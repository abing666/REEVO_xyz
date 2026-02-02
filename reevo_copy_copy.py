from typing import Optional, List, Dict, Tuple, Any
import logging
import subprocess
import numpy as np
import os
import glob
import random
import re
import signal
from omegaconf import DictConfig
import multiprocessing
import concurrent.futures
import psutil
import importlib.util
import traceback
import sys
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from string import Template
from utils.utils import *
from utils.llm_client.base import BaseClient
import atexit
import problems.cec2013.eval_copy as eval 

# ==========================================
# 辅助函数
# ==========================================

def save_solves(que, individual):
    # 主进程按组杀
    os.setsid()
    try:
        is_fast = individual.get('fast_mode', False)
        # 执行代码获取分数
        individual['obj'] = eval.solves(individual['code'], is_train=True,fast_mode=is_fast)
        individual['exec_success'] = True
        individual['traceback_msg'] = None
        logging.info(f"ID {individual['response_id']} 成功 | Obj: {individual['obj']}")
        with open(individual['stdout_filepath'], 'a', encoding='utf-8') as f_stdout:
            f_stdout.write(f"\n ID {individual['response_id']} 成功 | Obj: {individual['obj']}\n")
    except Exception:
        individual['obj'] = float(0)
        individual['exec_success'] = False
        individual['traceback_msg'] = traceback.format_exc()
        logging.error(f"ID {individual['response_id']} 失败 | Error: {individual['traceback_msg'].splitlines()[-1]}")
        with open(individual['stdout_filepath'], 'a', encoding='utf-8') as f_stdout:
            f_stdout.write(f"\n ID {individual['response_id']} 失败 | Error: {individual['traceback_msg']}\n")
    finally:
        # 把修改后的 individual 塞回队列
        que.put(individual)
        sys.exit(0)

# ==========================================
# 策略池核心架构 (新增)
# ==========================================

class StrategyNode:
    """策略节点：存储代码和统计信息"""
    def __init__(self, role: str, code: str, parent_id: Optional[str] = None):
        self.id: str = f"{role}_{str(uuid.uuid4())[:8]}" 
        self.role: str = role
        self.code: str = code
        self.parent_id: str = parent_id
        self.created_at: str = datetime.now().isoformat()
        
        self._stats = {
            "test_count": 0,
            "mean_score": 0.0,
            "m2": 0.0,
            "best_score": -1.0
        }

    def update_stats(self, score: float):
        self._stats["test_count"] += 1
        n = self._stats["test_count"]
        delta = score - self._stats["mean_score"]
        self._stats["mean_score"] += delta / n
        delta2 = score - self._stats["mean_score"]
        self._stats["m2"] += delta * delta2
        if score > self._stats["best_score"]:
            self._stats["best_score"] = score

    @property
    def metrics(self) -> Dict[str, float]:
        count = self._stats["test_count"]
        variance = self._stats["m2"] / count if count > 1 else 0.0
        return {
            "n": count,
            "mu": self._stats["mean_score"],
            "sigma": np.sqrt(variance),
            "max": self._stats["best_score"]
        }

    def __repr__(self):
        m = self.metrics
        return f"<{self.id} | µ={m['mu']:.3f}, n={m['n']}>"

@dataclass
class ExperimentRecord:
    combination_ids: Tuple[str, str, str]
    codes: Tuple[str, str, str]
    score: float = 0.0
    success: bool = False

class EvolutionaryOptimizer:
    """策略池管理器"""
    def __init__(self):
        self.pools: Dict[str, Dict[str, StrategyNode]] = {
            "slot_A": {}, "slot_B": {}, "slot_C": {}
        }
        self.synergy_tensor: Dict[Tuple[str, str, str], float] = {}

    def register_strategy(self, role: str, code: str, parent_id: str = None) -> str:
        if role not in self.pools:
            raise ValueError(f"Invalid role: {role}")
        node = StrategyNode(role, code, parent_id)
        self.pools[role][node.id] = node
        return node.id

    def sample_task(self, exploration_rate: float = 0.2) -> ExperimentRecord:
        """Epsilon-Greedy 采样"""
        selected_ids = []
        selected_codes = []
        roles = ["slot_A", "slot_B", "slot_C"]

        for r in roles:
            if not self.pools[r]:
                raise RuntimeError(f"Pool {r} is empty! Cannot sample.")

        for role in roles:
            candidates = list(self.pools[role].values())
            if random.random() < exploration_rate:
                node = random.choice(candidates)
            else:
                node = max(candidates, key=lambda x: x.metrics["mu"])
            selected_ids.append(node.id)
            selected_codes.append(node.code)

        return ExperimentRecord(combination_ids=tuple(selected_ids), codes=tuple(selected_codes))

    def commit_result(self, record: ExperimentRecord):
        """反向传播更新统计数据"""
        effective_score = record.score if record.success else 0.0
        self.synergy_tensor[record.combination_ids] = effective_score
        roles = ["slot_A", "slot_B", "slot_C"]
        for role, node_id in zip(roles, record.combination_ids):
            if node_id in self.pools[role]:
                self.pools[role][node_id].update_stats(effective_score)
    
    
    # ==========================================
    # 保存策略池到磁盘
    # ==========================================
    def save_pools(self, output_dir: str):
        """
        将策略池保存到指定文件夹 (TXT格式)。
        结构: 
        output_dir/
            slot_A/
                {id}.txt
            slot_B/
                ...
        """
        logging.info(f"Saving strategy pools to {output_dir}...")
        
        roles = ["slot_A", "slot_B", "slot_C"]
        
        for role in roles:
            # 创建子文件夹 (例如: 2026-xx-xx/strategy_pools/slot_A)
            role_dir = os.path.join(output_dir, role)
            os.makedirs(role_dir, exist_ok=True)
            
            pool = self.pools[role]
            for node_id, node in pool.items():
                file_path = os.path.join(role_dir, f"{node_id}.txt")
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        # 写入易读的文本头部信息
                        f.write(f"ID: {node.id}\n")
                        f.write(f"Role: {node.role}\n")
                        f.write(f"Parent ID: {node.parent_id}\n")
                        f.write(f"Created At: {node.created_at}\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"Metrics:\n")
                        f.write(f"  Test Count (n): {node.metrics['n']}\n")
                        f.write(f"  Mean Score (mu): {node.metrics['mu']:.4f}\n")
                        f.write(f"  Max Score: {node.metrics['max']:.4f}\n")
                        f.write(f"  Sigma: {node.metrics['sigma']:.4f}\n")
                        f.write("=" * 30 + " CODE " + "=" * 30 + "\n\n")
                        
                        # 写入代码正文
                        f.write(node.code)
                        
                except Exception as e:
                    logging.error(f"Failed to save strategy {node_id}: {e}")
                    
        logging.info("Strategy pools saved successfully.")

    def get_top_strategies(self, role: str, k: int) -> List[StrategyNode]:
        """
        获取指定角色中表现最好的前 k 个策略。
        """
        if role not in self.pools:
            return []
        
        # 获取所有策略节点
        nodes = list(self.pools[role].values())
        
        # 过滤掉没有被评估过的策略 (n=0)
        valid_nodes = [n for n in nodes if n.metrics["n"] > 0]
        
        # 如果没有有效策略，退而求其次取所有
        if not valid_nodes:
            valid_nodes = nodes

        # 按平均分 (mu) 降序排列
        valid_nodes.sort(key=lambda x: x.metrics["mu"], reverse=True)
        
        return valid_nodes[:k]

    # ==========================================
    # 策略池修剪功能
    # ==========================================
    def prune_pool(self, role: str, threshold: int, protected_ids: set = None):
        """
        当策略数量达到 threshold 时，删除一半表现最差的策略。
        保留依据：平均分 (mu) 最高的优先保留。
        """
        pool = self.pools[role]
        current_size = len(pool)
        
        # 没到阈值就不动
        if current_size < threshold:
            return

        logging.info(f"[Pruning] Pool {role} reached size {current_size} (Threshold: {threshold}). Executing purge...")

        # 1. 取出所有节点
        nodes = list(pool.values())
        
        # 2. 排序：按 mu (均分) 降序，分数高的排前面
        #    (如果分数相同，按测试次数 n 降序，优先保留经过更多验证的)
        nodes.sort(key=lambda x: (x.metrics["mu"], x.metrics["n"]), reverse=True)
        
        # 3. 确定保留数量：保留一半
        #    至少保留 2 个，避免池子被清空导致无法交叉
        # 保留一半
        keep_count = max(2, len(nodes) // 2)
        survivors = nodes[:keep_count]
        survivor_ids = {n.id for n in survivors}
        
        # 4. 打印被淘汰的最高分，看看是否误删了潜力股
        if len(nodes) > keep_count:
            best_rejected = nodes[keep_count]
            # logging.info(f"  -> Cutoff Score (mu): {best_rejected.metrics['mu']:.4f}")

         # 5. 强制拉回受保护的策略
        if protected_ids:
            for node in nodes[keep_count:]: # 检查被淘汰的
                if node.id in protected_ids and node.id not in survivor_ids:
                    survivors.append(node)
                    logging.info(f"Protected strategy {node.id} rescued from pruning.")

        # 6. 重建字典，只留下幸存者
        self.pools[role] = {n.id: n for n in survivors}
        
        logging.info(f"[Pruning] {role} size reduced: {current_size} -> {len(self.pools[role])}")

    # 在 EvolutionaryOptimizer 类中添加
    def get_diverse_strategies(self, role: str, k: int) -> List[StrategyNode]:
        """
        获取 k 个策略用于测试：
        1. 必须包含 1 个当前表现最好的策略 (Benchmark)。
        2. 其余 k-1 个从剩余策略中随机选择 (Robustness)。
        """
        if role not in self.pools:
            return []
        
        # 1. 获取所有有效的策略 (至少跑过一次的)
        pool_values = list(self.pools[role].values())
        valid_nodes = [n for n in pool_values if n.metrics["n"] > 0]
        
        # 如果没有有效策略（比如刚启动），尝试返回所有节点，或者空
        if not valid_nodes:
            if pool_values:
                # 还没有跑过分的策略，那就随机取
                return random.sample(pool_values, min(k, len(pool_values)))
            return []

        # 2. 找到精英 (Top 1)
        # 按 mu 降序排列，取第一个
        valid_nodes.sort(key=lambda x: x.metrics["mu"], reverse=True)
        elite_node = valid_nodes[0]
        
        result = [elite_node]
        
        # 3. 随机选取剩余的 k-1 个
        if k > 1:
            remaining_nodes = valid_nodes[1:] # 排除掉刚才选中的精英
            num_needed = k - 1
            
            if len(remaining_nodes) >= num_needed:
                # 池子够大，随机采样
                random_nodes = random.sample(remaining_nodes, num_needed)
            else:
                # 池子不够大，就全拿
                random_nodes = remaining_nodes
                
            result.extend(random_nodes)
            
        return result
# ==========================================
# ReEvo 主类
# ==========================================

class ReEvo:
    def __init__(
        self, 
        cfg: DictConfig, 
        root_dir: str, 
        generator_llm: BaseClient, 
        reflector_llm: Optional[BaseClient] = None,
        short_reflector_llm: Optional[BaseClient] = None,
        long_reflector_llm: Optional[BaseClient] = None,
        crossover_llm: Optional[BaseClient] = None,
        mutation_llm: Optional[BaseClient] = None
    ) -> None:
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.short_reflector_llm = short_reflector_llm or self.reflector_llm
        self.long_reflector_llm = long_reflector_llm or self.reflector_llm
        self.crossover_llm = crossover_llm or generator_llm
        self.mutation_llm = mutation_llm or generator_llm
        self.root_dir = root_dir
        
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.seed_funcs = [] 
        
        # 初始化优化器
        self.optimizer = EvolutionaryOptimizer()
        
        
        self.hydra_output_dir = os.getcwd() 
        save_path = os.path.join(self.hydra_output_dir, "codes")
        os.makedirs(save_path,exist_ok=True)
        logging.info(f"Real Output Directory captured: {self.hydra_output_dir}")
        # 只要 ReEvo 被实例化，无论何时程序退出，都会触发 self._save_on_exit
        atexit.register(self._save_on_exit)
        
        self.init_prompt()
        self.init_population()

    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.func_name = self.cfg.problem.func_name
        self.problem_type = self.cfg.problem.problem_type
        
        self.prompt_dir = f"{self.root_dir}/prompts"
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        self.problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        
        # 读取骨架代码
        self.code_bone = file_to_string(f'{self.problem_prompt_path}/code_bone.txt')
        self.func_signature = file_to_string(f'{self.problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{self.problem_prompt_path}/func_desc.txt')
        
        # 添加快速测试代码，这三个方案是肯定能执行通过的
        self.anchor_A = file_to_string(f'{self.problem_prompt_path}/code_slotA.txt')
        self.anchor_B = file_to_string(f'{self.problem_prompt_path}/code_slotB.txt')
        self.anchor_C = file_to_string(f'{self.problem_prompt_path}/code_slotC.txt')
        
        # 加载初始生成 Prompt
        self.init_generation_prompt = file_to_string(f'{self.problem_prompt_path}/prompt0_init_generation.txt')
       

        # 加载其他 Prompts (Reflection, Crossover, etc.)
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.user_reflector_lt_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_lt.txt')
        self.mutation_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        
        # Short term reflection prompts
        if os.path.exists(f'{self.prompt_dir}/common/user_reflector_st_bad.txt'):
            self.user_reflector_st_bad_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st_bad.txt')
        else:
            self.user_reflector_st_bad_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st.txt')

        if os.path.exists(f'{self.prompt_dir}/common/user_reflector_st_good.txt'):
            self.user_reflector_st_good_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st_good.txt')
        else:
            self.user_reflector_st_good_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st.txt')

        # Crossover prompts
        if os.path.exists(f'{self.prompt_dir}/common/crossover_bad.txt'):
            self.crossover_bad_prompt = file_to_string(f'{self.prompt_dir}/common/crossover_bad.txt')
        else:
            self.crossover_bad_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')

        if os.path.exists(f'{self.prompt_dir}/common/crossover_good.txt'):
            self.crossover_good_prompt = file_to_string(f'{self.prompt_dir}/common/crossover_good.txt')
        else:
            self.crossover_good_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')

        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name, 
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )

        self.print_crossover_prompt = True 
        self.print_mutate_prompt = True 
        self.print_short_term_reflection_prompt = True 
        self.print_long_term_reflection_prompt = True

    # ==========================================
    # 核心逻辑: 解析与组装 (New)
    # ==========================================

    def response_to_slot(self, response: str) -> List[str]:
        """
        步骤1: 解析 LLM 输出，将代码片段注册到策略池 (Ingestion)
        返回: 注册成功的 ID 列表
        """
        created_ids = []
        # 正则匹配 # BEGIN: ... Strategy X ... # END: ...
        pattern = re.compile(
            r"# BEGIN:.*?(Strategy [ABC]).*?\n(.*?)# END:.*?(?:Strategy [ABC].*?)?", 
            re.DOTALL | re.IGNORECASE
        )
        matches = list(pattern.finditer(response)) # 此时并未验证
        
        role_map = {"STRATEGY A": "slot_A", "STRATEGY B": "slot_B", "STRATEGY C": "slot_C"}
        
        # --- 1. 收集所有候选者 ---
        candidates_inds = []
        for match in matches:
            role_str = match.group(1).strip()
            code_body = match.group(2).strip()
            internal_role = role_map.get(role_str.upper())
            
            if internal_role and code_body:
                # 制作待测个体（只生成文件，不运行）
                ind = self._prepare_validation_ind(internal_role, code_body)
                if ind:
                    candidates_inds.append(ind)
        
        if not candidates_inds:
            return []

        # --- 2. 批量并行评估 (利用 evaluate_population 的多进程能力) ---
        logging.info(f"[Validation] Batch validating {len(candidates_inds)} strategies...")
        try:
            # 这里会一次性启动多个进程并行跑
            evaluated_inds = self.evaluate_population(candidates_inds)
            
            # --- 3. 处理结果并注册 ---
            for res_ind in evaluated_inds:
                role = res_ind["_meta_role"]
                code = res_ind["_meta_code"]
                
                if res_ind["exec_success"]:
                    try:
                        new_id = self.optimizer.register_strategy(role, code)
                        created_ids.append(new_id)
                        logging.info(f"[Validation Passed] Registered {new_id}")
                    except Exception as e:
                        logging.error(f"Strategy registration failed: {e}")
                else: 
                    # --- 失败的情况：保存“遗体” ---
                    error_msg = res_ind.get("traceback_msg", "Unknown Error")
                    
                    # 1. 保存到 error 文件夹
                    self._save_failed_strategy(role, code, error_msg)
                    
                    # 2. 日志提示
                    short_err = error_msg.strip().split('\n')[-1] if error_msg else "None"
                    logging.warning(f"[Validation Failed] Saved to {role}_error: {short_err}")

        finally:
            # --- 4. 批量清理临时文件 ---
            for ind in candidates_inds:
                if os.path.exists(ind["file_path"]): 
                    os.remove(ind["file_path"])
                if os.path.exists(ind["stdout_filepath"]): 
                    os.remove(ind["stdout_filepath"])
                    
        return created_ids

    # ==========================================
    # 快速测试LLM生成的一个策略是否可行
    # ==========================================
    def _prepare_validation_ind(self, role: str, code: str) -> dict:
        """辅助函数：只负责组装代码和写入临时文件，不运行评估"""
        # 1. 准备积木
        slotA = code if role == "slot_A" else self.anchor_A
        slotB = code if role == "slot_B" else self.anchor_B
        slotC = code if role == "slot_C" else self.anchor_C
        
        # 2. 拼接
        try:
            full_code = Template(self.code_bone).substitute(slotA=slotA, slotB=slotB, slotC=slotC)
        except Exception as e:
            logging.warning(f"[Validation] Template assembly failed: {e}")
            return None

        # 3. 构造临时文件和对象
        temp_id = f"val_{uuid.uuid4().hex[:8]}"
        file_path = f"codes/temp_{temp_id}.py"
        stdout_path = f"codes/temp_{temp_id}_stdout.txt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
            
        return {
            "file_path": file_path,
            "stdout_filepath": stdout_path,
            "code": full_code, 
            "code_run": full_code,
            "response_id": temp_id,
            "obj": 0.0,
            "exec_success": False,
            # 携带原始信息方便后续注册
            "_meta_role": role,
            "_meta_code": code,
            "fast_mode": True #快速测评使用，给eval模块降低资源
        }


    # 后续可以把这个选择改成MCTS选择
    # 这儿是把已经选好的三个策略组合起来
    def assemble_individual(self, id_a: str, id_b: str, id_c: str, source_info: str) -> dict:
        """
        步骤2: 根据 ID 从池子里取出代码，拼接成完整个体 (Assembly)
        """
        # 1. 获取节点
        node_a = self.optimizer.pools["slot_A"].get(id_a)
        node_b = self.optimizer.pools["slot_B"].get(id_b)
        node_c = self.optimizer.pools["slot_C"].get(id_c)
        
        if not (node_a and node_b and node_c):
            logging.error(f"Missing strategies for assembly: {id_a}, {id_b}, {id_c}")
            return None

        # 2. 准备代码
        slotA, slotB, slotC = node_a.code, node_b.code, node_c.code
        
        # 3. 模板拼接
        try:
            full_code = Template(self.code_bone).substitute(slotA=slotA, slotB=slotB, slotC=slotC)
            code_bone_run_str = file_to_string(f'{self.problem_prompt_path}/code_bone_run.txt')
            full_code_run = Template(code_bone_run_str).substitute(slotA=slotA, slotB=slotB, slotC=slotC)
        except Exception as e:
            logging.error(f"Code assembly failed: {e}")
            return None

        # 4. 生成元数据
        short_id = f"{id_a[-4:]}_{id_b[-4:]}_{id_c[-4:]}"
        response_id = f"iter{self.iteration}_{source_info}_{short_id}"
        file_path = f"codes/problem_iter{self.iteration}_{response_id}.txt"
        stdout_path = f"codes/problem_iter{self.iteration}_{response_id}_stdout.txt"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_code)

        return {
            "file_path": file_path,
            "stdout_filepath": stdout_path,
            "code": full_code,
            "code_run": full_code_run,
            "code_bone": self.code_bone,
            "slot_A": slotA,
            "slot_B": slotB,
            "slot_C": slotC,
            "strategy_ids": (id_a, id_b, id_c), # 关键：保存ID用于回写
            "response_id": response_id,
            "obj": 0.0,
            "exec_success": False,
            "error_msg": None,
            "fast_mode": False
        }

    def _update_optimizer_with_population(self, population: list[dict]):
        """辅助函数：将评估结果写回 Optimizer"""
        for ind in population:
            if "strategy_ids" in ind and ind["exec_success"]:
                record = ExperimentRecord(
                    combination_ids=ind["strategy_ids"],
                    codes=(ind["slot_A"], ind["slot_B"], ind["slot_C"]),
                    score=ind["obj"],
                    success=True
                )
                self.optimizer.commit_result(record)

    def _save_on_exit(self):
        """
        程序退出时的兜底保存逻辑。
        无论是在 init_population 还是 evolve 阶段退出，都会执行。
        """
        logging.info("Program exiting. Attempting to save strategy pools...")
        if hasattr(self, 'optimizer') and self.optimizer:
            # 确保路径存在
            current_work_dir = os.getcwd() 
            save_path = os.path.join(self.hydra_output_dir, "strategy_pools")
            self.optimizer.save_pools(save_path)
            
            abs_path = os.path.abspath(save_path)
            logging.info(f"✅ Strategy pools saved to: {abs_path}")
            print(f"Strategy pools saved to: {abs_path}") # print 确保在控制台也能看见
                
    def _save_failed_strategy(self, role: str, code: str, error_msg: str):
        """
        保存验证失败的策略，用于 Debug 分析。
        保存路径: outputs/.../strategy_pools/{role}_error/failed_{uuid}.txt
        """
        try:
            # 1. 构造专门存放错误的目录
            # 例如: .../strategy_pools/slot_A_error
            error_dir = os.path.join(self.hydra_output_dir, "strategy_pools", f"{role}_error")
            os.makedirs(error_dir, exist_ok=True)
            
            # 2. 构造唯一文件名
            unique_id = uuid.uuid4().hex[:8]
            file_path = os.path.join(error_dir, f"failed_{unique_id}.txt")
            
            # 3. 写入详细的“尸检报告”
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Role: {role}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write("=" * 30 + " ERROR TRACEBACK " + "=" * 30 + "\n")
                f.write(f"{error_msg}\n")
                f.write("=" * 30 + " ORIGINAL CODE " + "=" * 30 + "\n")
                f.write(code)
                
            # logging.debug(f"Saved failed strategy to {file_path}") # 调试用
            
        except Exception as e:
            logging.error(f"Failed to save error report: {e}")
               
    # ==========================================
    # 流程逻辑
    # ==========================================

    def init_population(self) -> None:
        """使用 LLM 生成初始策略池，并采样生成初始种群"""
        logging.info("Initializing population via LLM...")
        
        system = self.system_generator_prompt
        user_content = self.init_generation_prompt.format(
            code_bone=self.code_bone,
        )
        
        # 1. 请求 LLM 生成策略,生成init_pop_size个请求
        messages_lst = [[{"role": "system", "content": system}, {"role": "user", "content": user_content}] for _ in range(self.cfg.init_pop_size)]
        responses = self.generator_llm.multi_chat_completion(messages_lst, temperature=0.8)
        
        # 2. 解析并入库
        for response in responses:
            self.response_to_slot(response)
            
        # 3. 随机采样组装个体 (Initial Assembly)
        self.population = []
        for i in range(self.cfg.init_pop_size):
            try:
                # 初始阶段 exploration_rate=1.0 全随机
                # 这个sample task函数是算法关键，要想清楚怎么进行修改
                record = self.optimizer.sample_task(exploration_rate=1.0)
                ind = self.assemble_individual(*record.combination_ids, source_info=f"init_{i}")
                if ind: self.population.append(ind)
            except RuntimeError:
                break
                
        if not self.population:
            raise RuntimeError("Failed to assemble initial population. Pool might be empty.")

        # 4. 评估
        self.population = self.evaluate_population(self.population)
        
        # 5. 回写统计数据
        self._update_optimizer_with_population(self.population)
        
        # 6. 选出种子
        valid_seeds = [ind for ind in self.population if ind["exec_success"]]
        if not valid_seeds:
            raise RuntimeError("All initial individuals failed execution.")
            
        self.seed_ind = valid_seeds[0] # Best one as reference
        self.population = valid_seeds
        self.update_iter()

    def evaluate_population(self, population: list[dict]) -> list[dict]:
        """多进程评估"""
        
        # 1. 筛选需要评估的个体
        valid_inds = [ind for ind in population if ind["code_run"] is not None]
        if not valid_inds: return population
        # 2. 增加 evaluation 计数
        self.function_evals += len(valid_inds)
        # 3. 运行代码 (调用外部的 run_codes)
        # 建议 max_workers 设为 CPU核数/20，如果 eval 很重的话
        # 并行运行
        tasks_queue = list(valid_inds)
        running_tasks = []
        finished_individuals = []
        #创建进程
        # 同时只允许 8 个进程并行（模拟 CPU 核心数）
        MAX_CONCURRENT = 8
        TIMEOUT = self.cfg.timeout

        while tasks_queue or running_tasks:
            # --- A. 补充任务 ---
            while len(running_tasks) < MAX_CONCURRENT and tasks_queue:
                
                ind = tasks_queue.pop(0)
                # 1. 创建一个专属的“回信邮箱”
                q = multiprocessing.Queue()
                
                # 2. 启动进程，把 queue 和 individual 传过去
                p = multiprocessing.Process(target=save_solves, args=(q, ind))
                p.start()
                logging.info(f"ID {ind['response_id']} 开始执行... | PID: {p.pid}")
                
                running_tasks.append({
                    'process': p,
                    'queue': q, 
                    'start': time.time(), 
                    'data': ind # 存一份原始数据，万一超时了得用它来记录
                    })
                
            # --- B. 检查状态 ---
            for task in running_tasks[:]:
                p, q, start, data = task['process'], task['queue'], task['start'], task['data']
                elapsed = time.time() - start
                
                # 情况 1: 进程结束了 (可能是跑完了，也可能是报错被 try 捕获了)
                if not p.is_alive():
                    p.join()
                    
                    # 从邮箱里取回更新后的数据
                    if not q.empty():
                        finished_individuals.append(q.get())
                    else:
                        # 极其罕见：进程退出了但没写Queue (比如被操作系统OOM杀掉)
                        data['exec_success'] = False
                        finished_individuals.append(data)
                    running_tasks.remove(task)
                    continue
                # 情况 2: 超时 (Kill 之后也拿不到 Queue 数据了)
                if elapsed > TIMEOUT:
                    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL) # 发送 SIGKILL 信号
                    except: pass
                    p.join()
                    data['exec_success'] = False
                    finished_individuals.append(data)
                    running_tasks.remove(task)

        results_map = {i['response_id']: i for i in finished_individuals}
        for ind in population:
            if ind['response_id'] in results_map:
                res = results_map[ind['response_id']]
                ind.update({'obj': res['obj'], 'exec_success': res['exec_success'], 'traceback_msg': res.get('traceback_msg')})
        
        return population

    def update_iter(self) -> None:
        population = self.population
        objs = [ind["obj"] for ind in population]
        best_obj, best_idx = max(objs), np.argmax(objs)
        
        if self.best_obj_overall is None or best_obj < self.best_obj_overall: # Assuming min problem? Or max? Code says max(objs) above but usually minimizing. Using logic from previous code.
            # User's code used max(objs) for best_obj but checks < best_obj_overall (Minimization?). I will keep user's logic.
            # Assuming Minimization based on CEC2013 context usually
            # Wait, user code: if best_obj < self.best_obj_overall.
            # But best_obj = max(objs). This implies Maximization logic OR objs are negative errors.
            # I will strictly follow user's provided update_iter logic.
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_idx]["code"]
            self.best_code_path_overall = population[best_idx]["file_path"]

        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_idx]
        
        best_path = self.best_code_path_overall.replace(".py", ".txt").replace("code", "response")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {print_hyperlink(best_path, self.best_code_path_overall)}")
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Function Evals: {self.function_evals}")
            
        self.iteration += 1

    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def gen_st_reflection_prompts_split(self, ind1: dict, ind2: dict) -> tuple[list[dict], list[dict], str, str, float, float]:
        """
        Generate TWO separate prompts:
        1. Diagnose Worse Code (for Crossover Bad Task)
        2. Analyze Better Code (for Crossover Good Task)
        
        [Modified] Also return the scores (obj) of worse and better individuals.
        """
        if ind1["obj"] == ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        else: 
            better_ind, worse_ind = ind2, ind1

        worse_code = worse_ind["code"]
        better_code = better_ind["code"]
        seed_score = self.seed_ind["obj"]
        
        # [New] Extract scores
        worse_score = worse_ind["obj"]
        better_score = better_ind["obj"]
        seed_code_content = self.seed_ind["code"]
        
        system = self.system_reflector_prompt
        
        # --- Prompt 1: Bad Code Diagnosis ---
        user_bad = self.user_reflector_st_bad_prompt.format(
            func_name = self.func_name,
            func_desc = self.func_desc,
            problem_desc = self.problem_desc,
            worse_code=worse_code,
            better_code=better_code,
            score0 = worse_score,
            score1 = better_score,
            score2 = seed_score,
            seed_code = seed_code_content
        )
        msg_bad = [{"role": "system", "content": system}, {"role": "user", "content": user_bad}]
        
        # --- Prompt 2: Good Code Analysis ---
        user_good = self.user_reflector_st_good_prompt.format(
            func_name = self.func_name,
            func_desc = self.func_desc,
            problem_desc = self.problem_desc,
            worse_code=worse_code,
            better_code=better_code,
            score0 = worse_score,
            score1 = better_score,
            score2 = seed_score,
            seed_code = seed_code_content
        )
        msg_good = [{"role": "system", "content": system}, {"role": "user", "content": user_good}]

        # Print prompts for the first iteration (preview only bad, or both)
        if self.print_short_term_reflection_prompt:
            logging.info("ST Reflection BAD Prompt:\n" + user_bad)
            logging.info("ST Reflection GOOD Prompt:\n" + user_good)
            self.print_short_term_reflection_prompt = False
            
        return msg_bad, msg_good, worse_code, better_code, worse_score, better_score

    def short_term_reflection(self, population: list[dict]) -> tuple[list[str], list[str], list[str], list[str], list[float], list[float]]:
        """
        [Modified] Returns reflection lists AND score lists.
        """
        msgs_bad_lst = []
        msgs_good_lst = []
        worse_code_lst = []
        better_code_lst = []
        worse_score_lst = []
        better_score_lst = []
        
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Generate split prompts AND get scores
            msg_bad, msg_good, worse_code, better_code, worse_score, better_score = self.gen_st_reflection_prompts_split(parent_1, parent_2)
            
            msgs_bad_lst.append(msg_bad)
            msgs_good_lst.append(msg_good)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
            worse_score_lst.append(worse_score)
            better_score_lst.append(better_score)
        
        # Asynchronously generate responses
        logging.info(f"Generating {len(msgs_bad_lst)} Diagnoses & {len(msgs_good_lst)} Analyses...")
        
        responses_bad = self.short_reflector_llm.multi_chat_completion(msgs_bad_lst)
        responses_good = self.short_reflector_llm.multi_chat_completion(msgs_good_lst)
        
        return responses_bad, responses_good, worse_code_lst, better_code_lst, worse_score_lst, better_score_lst
    
    def long_term_reflection(self, reflections_bad: list[str], reflections_good: list[str]) -> None:
        """
        [Modified] Merge both types of reflections for long term memory.
        """
        # Combine reflections specifically
        combined_reflections = []
        for r_bad, r_good in zip(reflections_bad, reflections_good):
            combined_reflections.append(f"Diagnosis:\n{r_bad}\nAnalysis:\n{r_good}")

        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc = self.problem_desc,
            prior_reflection = self.long_term_reflection_str,
            new_reflection = "\n".join(combined_reflections),
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False
        
        self.long_term_reflection_str = self.long_reflector_llm.multi_chat_completion([messages])[0]
        
        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(combined_reflections) + '\n')
        
        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection_str + '\n')

    def select_strategy_parents(self, role: str, num_pairs: int = 1) -> list[list]:
        """
        从指定角色的策略池中随机选择策略对。
        特点：确保选出的两个策略性能不同，并按 [Better, Worse] 排序返回。
        
        Args:
            role: "slot_A", "slot_B", 或 "slot_C"
            num_pairs: 需要选出几对父代 (通常每次进化选 1 对即可)
            
        Returns:
            list: 包含配对策略节点的列表，例如 [[Best_Node, Worst_Node], ...]
        """
        # 1. 获取策略池中的所有候选者
        if role not in self.optimizer.pools:
            logging.error(f"Role {role} not found in pools.")
            return []
            
        pool = self.optimizer.pools[role]
        candidates = list(pool.values())
        
        # 2. 筛选有效策略 (仿照原代码的 eliminate invalid)
        # 这里的标准是：必须被测试过至少一次 (n > 0)
        valid_candidates = [
            node for node in candidates 
            if node.metrics["n"] > 0 and node.metrics["mu"] is not None
        ]
        
        # 如果池子太小，无法配对
        if len(valid_candidates) < 2:
            logging.warning(f"Strategy pool {role} has fewer than 2 valid candidates. Skipping crossover.")
            return []

        selected_pairs = []
        trial = 0
        
        # 3. 循环抽取 (仿照 while len < target)
        while len(selected_pairs) < num_pairs:
            trial += 1
            
            # 随机选择 2 个不重复的策略
            # 注意：用 random.sample 比 np.random.choice 对自定义对象更友好
            pair = random.sample(valid_candidates, 2)
            p1, p2 = pair[0], pair[1]
            
            # 4. 差异性检查 (仿照 if parents[0]["obj"] != parents[1]["obj"])
            # 我们比较平均分 mu，确保它们不是表现完全一样的策略（避免无效交叉）
            if abs(p1.metrics["mu"] - p2.metrics["mu"]) > 1e-9:
                # 5. 自动排序：[Better, Worse]
                # 这样正好对应提示词里的 Parent 1 (Better) 和 Parent 2 (Worse)
                if p1.metrics["mu"] > p2.metrics["mu"]:
                    selected_pairs.append([p1, p2]) 
                else:
                    selected_pairs.append([p2, p1]) 
            
            # 防止死循环 (当所有策略分数都一样时)
            if trial > 1000:
                logging.warning(f"Could not find enough distinct pairs in {role} after 1000 trials.")
                break
                
        return selected_pairs

    # crossed策略应该放到策略池，再从策略池选择，作为交叉下代
    # 辅助函数：处理 Crossover 结果
    def process_cross_response(self, response, idx, source_tag):
        """
        解析 LLM 响应，并将新策略与池中 [1个精英 + N个随机] 策略进行组合测试。
        """
        import itertools
        
        # A. 解析并入库
        new_ids = self.response_to_slot(response)
        if not new_ids: return []
        
        # B. 识别新生成的 ID
        new_id_map = {}
        for nid in new_ids:
            if "slot_A" in nid: new_id_map["slot_A"] = nid
            if "slot_B" in nid: new_id_map["slot_B"] = nid
            if "slot_C" in nid: new_id_map["slot_C"] = nid
            
        # C. 准备陪练：Top 1 + Random (N-1)
        TEST_PARTNER_LIMIT = 2 # 总共测试 2*2 次 (1个最好的 + 1个随机的)
        candidates = {"slot_A": [], "slot_B": [], "slot_C": []}
        roles = ["slot_A", "slot_B", "slot_C"]
        
        for role in roles:
            if role in new_id_map:
                # 主角：只测新生成的这个
                candidates[role] = [new_id_map[role]]
            else:
                # 配角：使用混合采样策略
                # 【修改点】调用 get_diverse_strategies
                nodes = self.optimizer.get_diverse_strategies(role, k=TEST_PARTNER_LIMIT)
                
                if nodes:
                    candidates[role] = [n.id for n in nodes]
                else:
                    # 兜底：如果池子完全是空的（极少见），用种子
                    seed_idx = roles.index(role)
                    seed_id = self.seed_ind["strategy_ids"][seed_idx]
                    candidates[role] = [seed_id]

        # D. 生成组合并组装
        # 笛卡尔积组合
        combinations = list(itertools.product(
            candidates["slot_A"], 
            candidates["slot_B"], 
            candidates["slot_C"]
        ))
        
        logging.info(f"Generating {len(combinations)} diverse test cases for {source_tag}_{idx}...")

        assembled_inds = []
        for i, (id_a, id_b, id_c) in enumerate(combinations):
            ind = self.assemble_individual(
                id_a, id_b, id_c, 
                source_info=f"{source_tag}_{idx}_v{i}"
            )
            if ind: 
                assembled_inds.append(ind)
                
        return assembled_inds

    def crossover(self) -> list[dict]:
        """
        按策略交叉，给B策略多一点资源 (0.25 : 0.5 : 0.25)
        """
        # 1. 计算各策略生成数量
        rateA, rateB, rateC = 0.25, 0.5, 0.25
        # 确保至少生成 1 个，避免 pop_size 很小时为 0
        pop_size = self.cfg.pop_size
        numsA = max(1, int(pop_size * rateA))
        numsB = max(1, int(pop_size * rateB))
        numsC = max(1, int(pop_size * rateC))

        messages = []
        system = self.system_generator_prompt
        problem_prompt_path = self.problem_prompt_path
        
        # 定义角色和数量的对应关系
        tasks = [
            ("slot_A", numsA),
            ("slot_B", numsB),
            ("slot_C", numsC)
        ]

        # 2. 准备 Prompts
        # 必须把这一步放在循环里，确保 slot_B 用的是 slot_B 的父代，而不是 slot_A 的
        for role, count in tasks:
            # 根据当前 role 选择父代，而不是写死 "slot_A"
            current_pairs = self.select_strategy_parents(role, count)
            
            if not current_pairs:
                logging.warning(f"No parents selected for {role}, skipping...")
                continue

            # 读取对应的 Prompt 模板
            prompt_path = f'{problem_prompt_path}/prompt1_crossover_{role}.txt'
            # 容错：如果特定角色的 prompt 不存在，尝试用通用的

            prompt_template = file_to_string(prompt_path)

            for better, worse in current_pairs:
                # 构造 User Content
                prompt_crossover = prompt_template.format(
                    code_bone = self.code_bone,
                    score_better = better.metrics["mu"],
                    code_better = better.code,
                    score_worse = worse.metrics["mu"],
                    code_worse = worse.code,
                )
                messages.append([
                    {"role": "system", "content": system}, 
                    {"role": "user", "content": prompt_crossover}
                ])
                
        logging.info(f"Generating {len(messages)} crossover codes...")
        
        if not messages:
            return []

        # 3. 获取 LLM 响应
        responses = self.crossover_llm.multi_chat_completion(
            messages, 
            temperature=self.crossover_llm.temperature
        )
        
        crossed_population = []

        # 4. 处理响应
        for i, resp in enumerate(responses):
            # 这里调用的 process_cross_response (它会返回一个列表，包含约9个个体)
            generated_inds_list = self.process_cross_response(resp, i, "cross")
            
            if generated_inds_list:
                # 这里必须用 extend，因为 generated_inds_list 是一个 list of dict
                # 如果用 append，crossed_population 会变成 [[ind1, ind2], [ind3...]]，导致后面 evaluate 报错
                crossed_population.extend(generated_inds_list)

        return crossed_population

    def mutate(self) -> list[dict]:
        """
        变异操作：基于当前各个角色的精英策略，生成新的变异策略。
        利用 process_cross_response 进行多重验证（N-Validation）。
        """
        # 1. 准备工作
        # 我们需要分别对 A, B, C 三个角色的精英进行变异
        roles = ["slot_A", "slot_B", "slot_C"]
        messages = []
        
        logging.info("Generating mutations based on Top strategies...")
        system = self.system_generator_prompt

        # 2. 构造变异 Prompt
        for role in roles:
            # 获取该角色当前最强的一个策略 (Top 1)
            # 注意：get_top_strategies 返回的是 list，我们取 [0]
            top_nodes = self.optimizer.get_top_strategies(role=role, k=1)
            
            if not top_nodes:
                logging.warning(f"No strategy found in pool {role} to mutate. Skipping.")
                continue
            
            elite_node = top_nodes[0]

            # 格式化 Prompt
            user_content = file_to_string(f'{self.problem_prompt_path}/prompt2_mutation_{role}.txt').format(
                code_bone = self.code_bone,
                score = elite_node.metrics["mu"], # 传入当前分数
                code = elite_node.code,           # 传入当前代码
            )
            messages.append([
                {"role": "system", "content": system}, 
                {"role": "user", "content": user_content}
            ])

        if not messages:
            return []

        # 3. 批量请求 LLM
        logging.info(f"Requesting {len(messages)} mutations via LLM...")
        # 变异建议使用较高的温度 (Temperature=1.0) 以激发创造力
        responses = self.mutation_llm.multi_chat_completion(messages, temperature=1.0)
        
        mutated_population = []
        
        # 4. 处理结果 (核心修改)
        for i, resp in enumerate(responses):
            # 直接复用 process_cross_response !
            # 它会自动解析新生成的代码 (e.g., New_A)，
            # 并自动去池子里找 [Top 1 + Random N] 的 B 和 C 进行组合测试。
            generated_inds_list = self.process_cross_response(resp, i, "mut")
            
            if generated_inds_list:
                # 注意：process_cross_response 返回的是 list，必须用 extend
                mutated_population.extend(generated_inds_list)

        logging.info(f"Generated {len(mutated_population)} mutated individuals (expanded combinations).")
        return mutated_population

    def evolve(self):
        POOL_LIMIT = 40
        max_iteration = 10
        while self.function_evals < self.cfg.max_fe or self.iteration<= max_iteration:
            # --- 1. 生成子代 ---
            # crossover 返回的是列表，建议这里命名为 offspring_crossed
            offspring_crossed = self.crossover()
            
            # --- 2. 评估子代 ---
            # 注意：evaluate_population 会原地修改 list 中的 dict，但也返回 list
            evaluated_crossed = self.evaluate_population(offspring_crossed)
            
            # --- 3. 变异 (Mutate) ---
            # 变异通常是基于当前精英进行的，生成额外的子代
            offspring_mutated = self.mutate()
            evaluated_mutants = self.evaluate_population(offspring_mutated)
            
            # --- 4. 合并种群 (父代 + 交叉子代 + 变异子代) ---
            # self.population 是上一代的幸存者
            combined_population = self.population + evaluated_crossed + evaluated_mutants
            
            # --- 5. 更新统计信息 ---
            # 把所有新评估过的个体注册到 Optimizer
            self._update_optimizer_with_population(evaluated_crossed + evaluated_mutants)
            
            # --- 6. 环境选择 (Survival Selection) ---
            # 筛选出执行成功的个体
            valid_pop = [ind for ind in combined_population if ind["exec_success"]]
            
            # 如果有效个体太少，为了防止种群灭绝，可以从历史最优或随机补全 (可选)
            if len(valid_pop) < self.cfg.pop_size:
                logging.warning("Valid population too small, keeping all valid ones.")
                next_gen = valid_pop
            else:
                # 截断选择：按分数降序排列，取前 pop_size 个
                # 假设是最大化问题 (best_obj = max(objs))
                valid_pop.sort(key=lambda x: x["obj"], reverse=True)
                next_gen = valid_pop[:self.cfg.pop_size]
            
            # 更新当前种群
            self.population = next_gen
            
            # --- 7. 更新全局状态 ---
            self.update_iter()
            
            # --- 8. 清理策略池 ---
            # 【重要】清理前需要获取当前种群正在使用的策略ID，防止把还在用的策略删了
            active_strategy_ids = set()
            for ind in self.population:
                if "strategy_ids" in ind:
                    active_strategy_ids.update(ind["strategy_ids"])
            
            # 如果有精英个体，也要保护
            if self.elitist and "strategy_ids" in self.elitist:
                active_strategy_ids.update(self.elitist["strategy_ids"])

            for role in ["slot_A", "slot_B", "slot_C"]:
                # 修改 prune_pool 接受 protected_ids
                self.optimizer.prune_pool(role, threshold=POOL_LIMIT, protected_ids=active_strategy_ids)