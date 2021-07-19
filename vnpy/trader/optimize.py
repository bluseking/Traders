from typing import Dict, List, Callable, Tuple
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from random import random, choice
from time import perf_counter
from multiprocessing import Manager, Pool, cpu_count

from deap import creator, base, tools
import numpy as np
import math

OUTPUT_FUNC = Callable[[str], None]
EVALUATE_FUNC = Callable[[dict], dict]
KEY_FUNC = Callable[[list], float]


# Create individual class used in genetic algorithm optimization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """
    Setting for runnning optimization.
    """

    def __init__(self) -> None:
        """"""
        self.params: Dict[str, List] = {}
        self.target_name: str = ""

    def add_parameter(
        self,
        name: str,
        start: float,
        end: float = None,
        step: float = None
    ) -> Tuple[bool, str]:
        """"""
        if end is None and step is None:
            self.params[name] = [start]
            return True, "固定参数添加成功"

        if start >= end:
            return False, "参数优化起始点必须小于终止点"

        if step <= 0:
            return False, "参数优化步进必须大于0"

        value: float = start
        value_list: List[float] = []

        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

        return True, f"范围参数添加成功，数量{len(value_list)}"

    def set_target(self, target_name: str) -> None:
        """"""
        self.target_name = target_name

    def generate_settings(self) -> List[dict]:
        """"""
        keys = self.params.keys()
        values = self.params.values()
        products = list(product(*values))

        settings = []
        for p in products:
            setting = dict(zip(keys, p))
            settings.append(setting)

        return settings


def check_optimization_setting(
    optimization_setting: OptimizationSetting,
    output: OUTPUT_FUNC = print
) -> bool:
    """"""
    if not optimization_setting.generate_settings():
        output("优化参数组合为空，请检查")
        return False

    if not optimization_setting.target_name:
        output("优化目标未设置，请检查")
        return False

    return True


def run_bf_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int = None,
    output: OUTPUT_FUNC = print
) -> List[Tuple]:
    """Run brutal force optimization"""
    settings: List[Dict] = optimization_setting.generate_settings()

    output(f"开始执行穷举算法优化")
    output(f"参数优化空间：{len(settings)}")

    start: int = perf_counter()

    with ProcessPoolExecutor(max_workers) as executor:
        results: List[Tuple] = list(executor.map(evaluate_func, settings))
        results.sort(reverse=True, key=key_func)

        end: int = perf_counter()
        cost: int = int((end - start))
        output(f"穷举算法优化完成，耗时{cost}秒")

        return results


def run_ga_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int = cpu_count() - 1,
    population_size: int = 50,
    ngen_size: int = 100,
    output: OUTPUT_FUNC = print
) -> List[Tuple]:
    """Run genetic algorithm optimization"""
    # Define functions for generate parameter randomly
    buf: List[Dict] = optimization_setting.generate_settings()
    settings: List[Tuple] = [list(d.items()) for d in buf]

    def generate_parameter() -> list:
        """"""
        return choice(settings)

    def mutate_individual(individual: list, indpb: float) -> tuple:
        """"""
        size = len(individual)
        paramlist = generate_parameter()
        for i in range(size):
            if random() < indpb:
                individual[i] = paramlist[i]
        return individual,

    results: list = []

    # Set up multiprocessing Pool and Manager
    with Manager() as manager, Pool(max_workers) as pool:
        # Create shared dict for result cache
        cache: Dict[Tuple, Tuple] = manager.dict()

        # Set up toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=1)
        toolbox.register("select", tools.selTournament, tournsize=2)

        toolbox.register("map", pool.map)
        toolbox.register(
            "evaluate",
            ga_evaluate,
            cache,
            evaluate_func,
            key_func
        )

        total_size: int = len(settings)
        pop_size: int = population_size         # population_size                      # number of individuals in each generation
        mu: int = int(4 + 3 * math.log(pop_size))
        lambda_: int = int(mu / 2)                              # number of children to produce at each generation
        # mu: int = int(pop_size * 0.8)                        # number of individuals to select for the next generation
        # 自定义算法中采用了动态交叉概率和动态变异概率，这里的设置做为速度模式初始值
        cxpb: float = 0.5       # 0.95         # probability that an offspring is produced by crossover
        mutpb: float = 0.0001      # 1 - cxpb    # probability that an offspring is produced by mutation
        dynamic_probability = 1             # 测试结果：启用动态概率会提高收敛的质量和平滑度，提高鲁棒性，但会延长一部分收敛的时间。
        dynamic_stop = 1                       # 增加停止的设定，达到精准度后自动停止，减少时间消耗。
        ngen: int = ngen_size    # number of generation

        pop: list = toolbox.population(pop_size)

        # Run ga optimization
        output(f"开始执行遗传算法优化")
        output(f"参数优化空间：{total_size}")
        output(f"每代族群总数：{pop_size}")
        output(f"优良筛选个数：{mu}")
        output(f"迭代次数：{ngen}")
        output(f"交叉概率：{cxpb:.0%}")
        output(f"突变概率：{mutpb:.0%}")
        output(f"动态调整双概率：{dynamic_probability}")
        output(f"动态迭代次数：{dynamic_stop}")
        start: int = perf_counter()

        def GA_accuracy(
            pop,
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            dynamic_probability=0
        ):
            # 数据记录
            npop = 100

            # 动态概率参数
            k1, k2, k3, k4 = 0.85, 0.5, 1.0, 0.05
            stats = tools.Statistics(key=lambda ind: ind.fitness.values)

            stats.register('avg', np.mean)
            stats.register('min', np.min)
            stats.register('max', np.max)
            stats.register('std', np.std)

            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (stats.fields)

            # 实现遗传算法
            # 评价族群
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # 记录数据
            record = stats.compile(pop)
            logbook.record(gen=0, nevals=len(invalid_ind), **record)
            states_step = []
            for gen in range(1, ngen + 1):
                # 配种选择
                offspring = toolbox.select(pop, 2 * npop)
                offspring = [toolbox.clone(_) for _ in offspring]  # 复制，否则在交叉和突变这样的原位操作中，会改变所有select出来的同个体副本
                states_step += [gen]
                # 变异操作 - 交叉
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if(dynamic_probability == 1):
                        # 生成动态交叉概率
                        max_child = np.max([child1.fitness.values, child2.fitness.values])
                        temp_max = logbook.select('max')[0]
                        temp_avg = logbook.select('avg')[0]
                        # print(temp_max,temp_avg)
                        # time.sleep(10)
                        if(max_child >= temp_avg):
                            cxpb = k1 * (temp_max - max_child) / (temp_max - temp_avg)
                            if cxpb <= 0:
                                cxpb = 0
                        else:
                            cxpb = k3
                        output(f"动态交叉概率：{cxpb:.0%}")
                    if random() < cxpb:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                # 变异操作 - 突变
                for mutant in offspring:
                    if (dynamic_probability == 1):
                        # 生成动态突变概率
                        temp_max = logbook.select('max')[0]
                        temp_avg = logbook.select('avg')[0]
                        if (mutant.fitness.values >= temp_avg):
                            mutpb = (k2 * (temp_max - mutant.fitness.values) / (temp_max - temp_avg))[0]
                            if mutpb <= 0:
                                mutpb = 0
                        else:
                            mutpb = k4
                        # print(mutpb)
                        # time.sleep(10)
                        output(f"动态突变概率：{mutpb:.0%}")
                    if random() < mutpb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                # 评价当前没有fitness的个体，确保整个族群中的个体都有对应的适应度
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                # 环境选择 - 保留精英
                pop = tools.selBest(offspring, npop, fit_attr='fitness')  # 选择精英,保持种群规模
                pop[:] = offspring      # 重插入操作

                # 记录数据
                record = stats.compile(pop)
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                # 动态停止迭代
                if(dynamic_stop == 1 and logbook.select('std')[0] <= (10**-10)):
                    return pop, logbook
                    break
            return pop, logbook

        resule_GA, logbook_GA = GA_accuracy(
            pop,
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            dynamic_probability=1,
        )
        end: int = perf_counter()
        cost: int = int((end - start))

        output(f"遗传算法优化完成，耗时{cost}秒")

        results: list = list(cache.values())
        results.sort(reverse=True, key=key_func)
        return results, logbook_GA


def ga_evaluate(
    cache: dict,
    evaluate_func: callable,
    key_func: callable,
    parameters: list
) -> float:
    """
    Functions to be run in genetic algorithm optimization.
    """
    tp: tuple = tuple(parameters)
    if tp in cache:
        result: tuple = cache[tp]
    else:
        setting: dict = dict(parameters)
        result: dict = evaluate_func(setting)
        cache[tp] = result

    value: float = key_func(result)
    return (value, )
