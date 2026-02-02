# REEVO_xyz

1、初始化：
提示词：（10份一样的提示词）prompt0_init_generation.txt，code_bone.txt
快速验证：返回A_LLM+B_LLM+C_LLM。快速测评A_LLM+B_fbkde+C_fbkde，如果能执行，则将A_LLM保存到策略池中，否则丢弃。
初始化种群构建：从策略池中随机采样，得到10份A_rand+B_rand+C_rand，测评这十个代码，并按如下公式更新对应策略分数，
delta = score - A_rand["mean_score"]
A_rand["mean_score"] += delta / n
即认为代码分数是每个策略的得分。

2、交叉
提示词：（8次，交叉一次A、C，交叉两次B）见附录b：prompt1_crossover_slot_A.txt, prompt1_crossover_slot_B.txt, prompt1_crossover_slot_C.txt
父代选择：从策略池中选择两个均值不一样的策略，将其填入交叉提示词模板。
LLM结果：新生成的策略与其他策略下的“精英+随机”进行笛卡尔积组合，生成4个体进行评估。

3、变异：
提示词：（3次，变异一次最好的A、B、C）见附录C：prompt2_mutation_slot_A.txt, prompt2_mutation_slot_B.txt, prompt2_mutation_slot_C.txt
结果：新生成的策略与其他策略下的“精英+随机”进行笛卡尔积组合，生成4个体进行评估。
