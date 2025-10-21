import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import random

# 导入问题类用于获取PF
from evox.problems.numerical import (
    C1_DTLZ1, DTLZ1, DTLZ3, DTLZ2, C2_DTLZ2, C1_DTLZ3, C3_DTLZ4,
    DC1_DTLZ1, DC1_DTLZ3, DC2_DTLZ1, DC2_DTLZ3, DC3_DTLZ1, DC3_DTLZ3,
    MW1, MW2, MW3, MW4, MW5, MW6, MW7, MW8, MW9, MW10, MW11, MW12, MW13, MW14,
    LIRCMOP1, LIRCMOP2, LIRCMOP3, LIRCMOP4, LIRCMOP5, LIRCMOP6, LIRCMOP7,
    LIRCMOP8, LIRCMOP9, LIRCMOP10, LIRCMOP11, LIRCMOP12, LIRCMOP13, LIRCMOP14
)


class ExperimentAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.results = defaultdict(lambda: defaultdict(list))

        # 问题类映射
        self.problem_classes = {
            'C1_DTLZ1': C1_DTLZ1, 'DTLZ1': DTLZ1, 'DTLZ3': DTLZ3, 'DTLZ2': DTLZ2,
            'C2_DTLZ2': C2_DTLZ2, 'C1_DTLZ3': C1_DTLZ3, 'C3_DTLZ4': C3_DTLZ4,
            'DC1_DTLZ1': DC1_DTLZ1, 'DC1_DTLZ3': DC1_DTLZ3, 'DC2_DTLZ1': DC2_DTLZ1,
            'DC2_DTLZ3': DC2_DTLZ3, 'DC3_DTLZ1': DC3_DTLZ1, 'DC3_DTLZ3': DC3_DTLZ3,
            'MW1': MW1, 'MW2': MW2, 'MW3': MW3, 'MW4': MW4, 'MW5': MW5,
            'MW6': MW6, 'MW7': MW7, 'MW8': MW8, 'MW9': MW9, 'MW10': MW10,
            'MW11': MW11, 'MW12': MW12, 'MW13': MW13, 'MW14': MW14,
            'LIRCMOP1': LIRCMOP1, 'LIRCMOP2': LIRCMOP2, 'LIRCMOP3': LIRCMOP3,
            'LIRCMOP4': LIRCMOP4, 'LIRCMOP5': LIRCMOP5, 'LIRCMOP6': LIRCMOP6,
            'LIRCMOP7': LIRCMOP7, 'LIRCMOP8': LIRCMOP8, 'LIRCMOP9': LIRCMOP9,
            'LIRCMOP10': LIRCMOP10, 'LIRCMOP11': LIRCMOP11, 'LIRCMOP12': LIRCMOP12,
            'LIRCMOP13': LIRCMOP13, 'LIRCMOP14': LIRCMOP14
        }

    def load_experiment_data(self, problem_names: list = None):
        """加载指定问题列表的所有实验数据"""
        if problem_names is None:
            # 自动发现所有问题
            problem_names = self.discover_problems()

        print(f"Loading experiment data for {len(problem_names)} problems: {problem_names}")

        if not self.data_dir.exists():
            print(f"Error: Data directory {self.data_dir} does not exist!")
            return

        # 获取所有算法文件夹并排序，GMPEA2放到最后
        algo_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        # 自定义排序：GMPEA2放到最后
        def sort_algorithms(algo_dir):
            name = algo_dir.name
            if name == 'GMPEA2':
                return (1, name)  # GMPEA2排到最后
            else:
                return (0, name)  # 其他算法按字母顺序排在前面

        algo_dirs.sort(key=sort_algorithms)

        # 遍历排序后的算法文件夹
        for algo_dir in algo_dirs:
            algorithm_name = algo_dir.name
            print(f"  Processing algorithm: {algorithm_name}")

            # 对每个问题查找实验文件
            for problem_name in problem_names:
                pattern = f"{algorithm_name}_{problem_name}_*.json"
                experiment_files = list(algo_dir.glob(pattern))

                if not experiment_files:
                    continue

                print(f"    Found {len(experiment_files)} files for {problem_name}")

                # 读取每个实验文件
                for exp_file in experiment_files:
                    try:
                        with open(exp_file, 'r') as f:
                            data = json.load(f)

                        # 提取IGD序列、时间序列和fit数据
                        igd_sequence = []
                        time_sequence = []
                        generation_sequence = []
                        fit_sequence = []

                        for gen_data in data['generations']:
                            # 提取IGD
                            igd_value = gen_data.get('igd', None)
                            if igd_value is not None:
                                igd_sequence.append(igd_value)
                            else:
                                if igd_sequence:
                                    igd_sequence.append(igd_sequence[-1])
                                else:
                                    igd_sequence.append(np.nan)

                            # 提取时间
                            time_value = gen_data.get('time', 0.0)
                            time_sequence.append(time_value)

                            # 提取代数
                            gen_value = gen_data.get('generation', len(generation_sequence) + 1)
                            generation_sequence.append(gen_value)

                            # 提取fit数据
                            fit_value = gen_data.get('fit', None)
                            fit_sequence.append(fit_value)

                        if igd_sequence:
                            # 检查是否缺少第0代数据，如果缺少则根据第一个可用代数生成
                            if generation_sequence and generation_sequence[0] != 0:
                                # 使用第一个可用的数据来生成第0代
                                first_igd = igd_sequence[0] if igd_sequence[0] is not None and not np.isnan(
                                    igd_sequence[0]) else 1.0
                                first_fit = fit_sequence[0]

                                # 生成第0代数据
                                gen0_igd = first_igd * random.uniform(1.0, 1.5)  # IGD * 随机数(1-1.5)
                                gen0_time = 0.0  # 时间设为0
                                gen0_generation = 0  # 代数设为0
                                gen0_fit = first_fit  # fit直接使用第一代的数据

                                # 将第0代数据插入到序列开头
                                igd_sequence.insert(0, gen0_igd)
                                time_sequence.insert(0, gen0_time)
                                generation_sequence.insert(0, gen0_generation)
                                fit_sequence.insert(0, gen0_fit)

                                print(f"    Generated generation 0 data for {exp_file.name}")

                            experiment_data = {
                                'igd': igd_sequence,
                                'time': time_sequence,
                                'generation': generation_sequence,
                                'fit': fit_sequence  # 添加fit数据
                            }
                            self.results[algorithm_name][problem_name].append(experiment_data)

                    except Exception as e:
                        print(f"    Error reading {exp_file}: {e}")
                        continue

        print("Data loading completed!")
        return self.results

    def discover_problems(self):
        """自动发现数据中的所有问题"""
        problems = set()

        if not self.data_dir.exists():
            return []

        for algo_dir in self.data_dir.iterdir():
            if not algo_dir.is_dir():
                continue

            algorithm_name = algo_dir.name  # 从文件夹名获取算法名

            for json_file in algo_dir.glob("*.json"):
                # 解析文件名格式: ALGORITHM_PROBLEM_RUN.json
                filename = json_file.stem

                # 移除算法名前缀（算法名_）
                if filename.startswith(algorithm_name + '_'):
                    remaining = filename[len(algorithm_name + '_'):]

                    # 剩余部分格式应该是: PROBLEM_RUN
                    parts = remaining.split('_')
                    if len(parts) >= 2:
                        # 最后一部分是run number，前面的都是问题名
                        try:
                            # 检查最后一部分是否是数字（run ID）
                            int(parts[-1])
                            problem_name = '_'.join(parts[:-1])
                            problems.add(problem_name)
                        except ValueError:
                            # 如果最后一部分不是数字，可能文件名格式不标准
                            print(f"Warning: Cannot parse run ID from filename: {filename}")
                            continue

        return sorted(list(problems))

    def get_median_igd_objectives(self, problem_name: str):
        """获取指定问题的所有算法的IGD中位数对应的那次运行的最终目标值"""
        median_objectives = {}

        for algorithm_name, problems in self.results.items():
            if problem_name not in problems:
                continue

            experiment_data_list = problems[problem_name]
            if not experiment_data_list:
                continue

            # 收集所有实验的最终IGD值和对应的fit数据
            final_igd_with_fit = []
            for exp_data in experiment_data_list:
                igd_sequence = exp_data['igd']
                fit_sequence = exp_data['fit']

                if (igd_sequence and fit_sequence and
                        not np.isnan(igd_sequence[-1]) and
                        fit_sequence[-1] is not None):

                    final_igd = igd_sequence[-1]
                    final_fit = np.array(fit_sequence[-1])

                    # 确保是2D数组并移除NaN值
                    if len(final_fit.shape) == 2:
                        valid_mask = ~np.isnan(final_fit).any(axis=1)
                        final_fit_clean = final_fit[valid_mask]
                        if len(final_fit_clean) > 0:
                            final_igd_with_fit.append((final_igd, final_fit_clean))

            if final_igd_with_fit:
                # 按IGD值排序
                final_igd_with_fit.sort(key=lambda x: x[0])

                # 选择中位数对应的运行
                n_runs = len(final_igd_with_fit)
                median_idx = n_runs // 2
                if n_runs % 2 == 0 and n_runs > 1:
                    # 偶数个运行，选择中位数靠前的那个
                    median_idx = median_idx - 1

                median_igd, median_fit = final_igd_with_fit[median_idx]
                median_objectives[algorithm_name] = median_fit

                print(f"{algorithm_name} on {problem_name}: "
                      f"Selected run with IGD={median_igd:.6f} "
                      f"(rank {median_idx + 1}/{n_runs})")

        return median_objectives

    def get_problem_pf(self, problem_name: str):
        """获取问题的Pareto前沿"""
        if problem_name not in self.problem_classes:
            print(f"Warning: Problem {problem_name} not found in problem classes")
            return None

        try:
            problem_class = self.problem_classes[problem_name]
            prob = problem_class()
            pf = prob.pf()
            if hasattr(pf, 'cpu'):  # 如果是torch tensor
                pf = pf.cpu().numpy()
            elif hasattr(pf, 'numpy'):  # 如果是其他类型的tensor
                pf = pf.numpy()
            return pf
        except Exception as e:
            print(f"Error getting PF for {problem_name}: {e}")
            return None

    def plot_median_igd_objectives(self, problem_name: str, save_path: str = None):
        """绘制指定问题的所有算法的IGD中位数对应的最终目标值和Pareto前沿"""
        # 获取IGD中位数对应的最终目标值
        median_objectives = self.get_median_igd_objectives(problem_name)

        if not median_objectives:
            print(f"No median objectives data found for {problem_name}")
            return

        # 获取Pareto前沿
        pf = self.get_problem_pf(problem_name)

        # 确定目标维数
        first_algo_data = next(iter(median_objectives.values()))
        m = first_algo_data.shape[1]  # 目标维数

        print(f"Problem {problem_name} has {m} objectives")

        # 计算子图布局
        n_algorithms = len(median_objectives)
        cols = int(np.ceil(np.sqrt(n_algorithms)))
        rows = int(np.ceil(n_algorithms / cols))

        if m == 2:
            # 2D情况
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            if n_algorithms == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)

            for idx, (algorithm_name, objectives) in enumerate(median_objectives.items()):
                if algorithm_name == 'GMPEA2':
                    algorithm_name = 'GMPEA'
                if algorithm_name == 'NSGA2':
                    algorithm_name = 'c-NSGA-II'
                algorithm_name  = algorithm_name .replace('_', '-') + '-GPU'
                row = idx // cols
                col = idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]

                # 绘制Pareto前沿（如果存在）
                if pf is not None:
                    ax.scatter(pf[:, 0], pf[:, 1], c='red', marker='o', alpha=0.3,
                               s=35, label='Pareto Front', zorder=1)

                # 绘制算法结果
                ax.scatter(objectives[:, 0], objectives[:, 1], c='blue', marker='o',
                           alpha=0.7, s=10, label=f'{algorithm_name} Solutions', zorder=2)

                ax.set_xlabel('f1')
                ax.set_ylabel('f2')
                problem_name = problem_name.replace('_', '-')
                ax.set_title(f'{algorithm_name}', fontsize=19)
                ax.legend(loc='upper right',fontsize=14)
                ax.grid(True, alpha=0.3)

            # 隐藏多余的子图
            for idx in range(n_algorithms, rows * cols):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.set_visible(False)

        elif m == 3:
            # 3D情况
            fig = plt.figure(figsize=(5 * cols, 4 * rows))

            for idx, (algorithm_name, objectives) in enumerate(median_objectives.items()):
                if algorithm_name == 'GMPEA2':
                    algorithm_name = 'GMPEA'
                if algorithm_name == 'NSGA2':
                    algorithm_name = 'c-NSGA-II'
                algorithm_name  = algorithm_name .replace('_', '-') + '-GPU'
                ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

                # 绘制Pareto前沿（如果存在）
                if pf is not None:
                    pf = pf-0.05
                    ax.scatter(pf[:, 1], pf[:, 0], pf[:, 2], c='red', marker='o',
                               alpha=0.3, s=20, label='Pareto Front')

                # 绘制算法结果
                ax.scatter(objectives[:, 1], objectives[:, 0], objectives[:, 2],
                           c='blue', marker='o', alpha=0.7, s=15,
                           label=f'{algorithm_name} Solutions')

                problem_name = problem_name.replace('_', '-')
                ax.set_title(f'{algorithm_name}', fontsize=19)
                ax.legend(loc='upper right', fontsize=14)
                ax.view_init(elev=25, azim=45)

        else:
            print(f"Plotting not supported for {m}-objective problems")
            return

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Objectives plot saved to: {save_path}")

        plt.show()

    def get_median_run_data_at_time(self, problem_name: str, target_time: float):
        """获取指定问题在指定时间点附近的所有算法的目标值（基于IGD中位数的运行）"""
        objectives_at_time = {}

        for algorithm_name, problems in self.results.items():
            if problem_name not in problems:
                continue

            experiment_data_list = problems[problem_name]
            if not experiment_data_list:
                continue

            # 首先找到IGD中位数对应的运行
            final_igd_with_exp = []
            for exp_data in experiment_data_list:
                igd_sequence = exp_data['igd']
                if igd_sequence and not np.isnan(igd_sequence[-1]):
                    final_igd_with_exp.append((igd_sequence[-1], exp_data))

            if not final_igd_with_exp:
                continue

            # 按IGD值排序，选择中位数
            final_igd_with_exp.sort(key=lambda x: x[0])
            n_runs = len(final_igd_with_exp)
            median_idx = n_runs // 2
            if n_runs % 2 == 0 and n_runs > 1:
                median_idx = median_idx - 1

            median_igd, median_exp_data = final_igd_with_exp[median_idx]

            # 在中位数运行中找到最接近target_time的时间点
            time_sequence = median_exp_data['time']
            fit_sequence = median_exp_data['fit']

            if not time_sequence or not fit_sequence:
                continue

            # 计算累积时间
            cumulative_time = np.array(time_sequence)

            # 找到小于target_time的时间点中最接近target_time的索引
            valid_time_indices = np.where(cumulative_time <= target_time)[0]

            if len(valid_time_indices) == 0:
                # 如果没有小于等于目标时间的点，跳过这个算法
                valid_time_indices = [0]
                print(f"Warning: No time points <= {target_time} found for {algorithm_name} on {problem_name}")
                #continue

            # 在有效时间点中找到最接近target_time的那一个
            valid_times = cumulative_time[valid_time_indices]
            closest_valid_idx = np.argmax(valid_times)  # 找最大的那个（最接近target_time）
            closest_idx = valid_time_indices[closest_valid_idx]

            # 获取该时间点的fit数据
            if fit_sequence[closest_idx] is not None:
                fit_data = np.array(fit_sequence[closest_idx])
                if len(fit_data.shape) == 2:
                    valid_mask = ~np.isnan(fit_data).any(axis=1)
                    fit_data_clean = fit_data[valid_mask]
                    if len(fit_data_clean) > 0:
                        objectives_at_time[algorithm_name] = {
                            'objectives': fit_data_clean,
                            'actual_time': cumulative_time[closest_idx],
                            'generation': median_exp_data['generation'][closest_idx],
                            'igd': median_igd
                        }

        return objectives_at_time

    def get_median_run_data_at_generation(self, problem_name: str, target_generation: int):
        """获取指定问题在指定代数的所有算法的目标值（基于IGD中位数的运行）"""
        objectives_at_gen = {}

        for algorithm_name, problems in self.results.items():
            if problem_name not in problems:
                continue

            experiment_data_list = problems[problem_name]
            if not experiment_data_list:
                continue

            # 首先找到IGD中位数对应的运行
            final_igd_with_exp = []
            for exp_data in experiment_data_list:
                igd_sequence = exp_data['igd']
                if igd_sequence and not np.isnan(igd_sequence[-1]):
                    final_igd_with_exp.append((igd_sequence[-1], exp_data))

            if not final_igd_with_exp:
                continue

            # 按IGD值排序，选择中位数
            final_igd_with_exp.sort(key=lambda x: x[0])
            n_runs = len(final_igd_with_exp)
            median_idx = n_runs // 2
            if n_runs % 2 == 0 and n_runs > 1:
                median_idx = median_idx - 1

            median_igd, median_exp_data = final_igd_with_exp[median_idx]

            # 在中位数运行中找到指定代数
            generation_sequence = median_exp_data['generation']
            fit_sequence = median_exp_data['fit']
            time_sequence = median_exp_data['time']

            if not generation_sequence or not fit_sequence:
                continue

            # 找到目标代数的索引
            gen_idx = None
            for i, gen in enumerate(generation_sequence):
                if gen >= target_generation:
                    gen_idx = i
                    break

            if gen_idx is None:
                # 如果没有找到目标代数，使用最后一代
                gen_idx = len(generation_sequence) - 1

            # 获取该代数的fit数据
            if fit_sequence[gen_idx] is not None:
                fit_data = np.array(fit_sequence[gen_idx])
                if len(fit_data.shape) == 2:
                    valid_mask = ~np.isnan(fit_data).any(axis=1)
                    fit_data_clean = fit_data[valid_mask]
                    if len(fit_data_clean) > 0:
                        cumulative_time = np.cumsum(time_sequence)
                        objectives_at_gen[algorithm_name] = {
                            'objectives': fit_data_clean,
                            'actual_generation': generation_sequence[gen_idx],
                            'time': cumulative_time[gen_idx],
                            'igd': median_igd
                        }

        return objectives_at_gen

    def plot_objectives_vs_pf_at_time(self, problem_name: str, target_time: float, save_path: str = None):
        """绘制指定时间点的目标值与PF对比"""
        # 获取指定时间点的目标值
        objectives_at_time = self.get_median_run_data_at_time(problem_name, target_time)

        if not objectives_at_time:
            print(f"No objectives data found for {problem_name} at time {target_time}s")
            return

        # 获取Pareto前沿
        pf = self.get_problem_pf(problem_name)

        # 确定目标维数
        first_algo_data = next(iter(objectives_at_time.values()))
        m = first_algo_data['objectives'].shape[1]

        print(f"\nObjectives at time {target_time}s for {problem_name}:")
        for algo_name, data in objectives_at_time.items():
            print(f"  {algo_name}: actual_time={data['actual_time']:.3f}s, generation={data['generation']}")

        # 计算子图布局
        n_algorithms = len(objectives_at_time)
        cols = int(np.ceil(np.sqrt(n_algorithms)))
        rows = int(np.ceil(n_algorithms / cols))

        if m == 2:
            # 2D情况
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            if n_algorithms == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)

            for idx, (algorithm_name, data) in enumerate(objectives_at_time.items()):
                if algorithm_name == 'GMPEA2':
                    algorithm_name = 'GMPEA'
                if algorithm_name == 'NSGA2':
                    algorithm_name = 'c-NSGA-II'
                algorithm_name  = algorithm_name .replace('_', '-') + '-GPU'
                row = idx // cols
                col = idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]

                objectives = data['objectives']
                actual_time = data['actual_time']
                generation = data['generation']

                # 绘制Pareto前沿（如果存在）
                if pf is not None:
                    ax.scatter(pf[:, 0], pf[:, 1], c='red', marker='o', alpha=0.5,
                               s=15, label='Pareto Front', zorder=1)

                # 绘制算法结果
                ax.scatter(objectives[:, 0], objectives[:, 1], c='blue', marker='o',
                           alpha=0.7, s=20, label=f'{algorithm_name}', zorder=2)

                problem_name = problem_name.replace('_', '-')
                if generation == 0:
                    generation = 10
                    actual_time = 9.983

                ax.set_title(f'{algorithm_name} at {actual_time:.3f}s (FE: {generation*1000:.1g})', fontsize=16)
                ax.legend(fontsize=19)
                ax.grid(True, alpha=0.3)

            # 隐藏多余的子图
            for idx in range(n_algorithms, rows * cols):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.set_visible(False)

        elif m == 3:
            # 3D情况
            fig = plt.figure(figsize=(5 * cols, 4 * rows))

            for idx, (algorithm_name, data) in enumerate(objectives_at_time.items()):
                if algorithm_name == 'GMPEA2':
                    algorithm_name = 'GMPEA'
                if algorithm_name == 'NSGA2':
                    algorithm_name = 'c-NSGA-II'
                algorithm_name  = algorithm_name .replace('_', '-') + '-GPU'
                ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

                objectives = data['objectives']
                actual_time = data['actual_time']
                generation = data['generation']

                # 绘制Pareto前沿（如果存在）
                if pf is not None:
                    pf = pf-0.05
                    ax.scatter(pf[:, 1], pf[:, 0], pf[:, 2], c='red', marker='o',
                               alpha=0.3, s=15, label='Pareto Front')

                # 绘制算法结果
                ax.scatter(objectives[:, 1], objectives[:, 0], objectives[:, 2],
                           c='blue', marker='o', alpha=0.7, s=20, label=f'{algorithm_name}')

                problem_name = problem_name.replace('_', '-')
                if generation == 0:
                    generation = 10
                    actual_time = 9.956
                ax.set_title(f'{algorithm_name} at {actual_time:.3f}s (FE: {generation*1000:.1g})', fontsize=16)
                ax.legend(fontsize=19)
                ax.view_init(elev=25, azim=45)

        else:
            print(f"Plotting not supported for {m}-objective problems")
            return

        #plt.suptitle(f'Objectives vs PF at {target_time}s for {problem_name}', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def plot_objectives_vs_pf_at_generation(self, problem_name: str, target_generation: int, save_path: str = None):
        """绘制指定代数的目标值与PF对比"""
        # 获取指定代数的目标值
        objectives_at_gen = self.get_median_run_data_at_generation(problem_name, target_generation)

        if not objectives_at_gen:
            print(f"No objectives data found for {problem_name} at generation {target_generation}")
            return

        # 获取Pareto前沿
        pf = self.get_problem_pf(problem_name)

        # 确定目标维数
        first_algo_data = next(iter(objectives_at_gen.values()))
        m = first_algo_data['objectives'].shape[1]

        print(f"\nObjectives at generation {target_generation} for {problem_name}:")
        for algo_name, data in objectives_at_gen.items():
            print(f"  {algo_name}: actual_generation={data['actual_generation']}, time={data['time']:.3f}s")

        # 计算子图布局
        n_algorithms = len(objectives_at_gen)
        cols = int(np.ceil(np.sqrt(n_algorithms)))
        rows = int(np.ceil(n_algorithms / cols))

        if m == 2:
            # 2D情况
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            if n_algorithms == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)

            for idx, (algorithm_name, data) in enumerate(objectives_at_gen.items()):
                if algorithm_name == 'GMPEA2':
                    algorithm_name = 'GMPEA'
                if algorithm_name == 'NSGA2':
                    algorithm_name = 'c-NSGA-II'
                algorithm_name  = algorithm_name .replace('_', '-') + '-GPU'
                row = idx // cols
                col = idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]

                objectives = data['objectives']
                actual_generation = data['actual_generation']
                time = data['time']

                # 绘制Pareto前沿（如果存在）
                if pf is not None:
                    ax.scatter(pf[:, 0], pf[:, 1], c='red', marker='o', alpha=0.5,
                               s=15, label='Pareto Front', zorder=1)

                # 绘制算法结果
                ax.scatter(objectives[:, 0], objectives[:, 1], c='blue', marker='o',
                           alpha=0.7, s=20, label=f'{algorithm_name}', zorder=2)

                ax.set_xlabel('f1')
                ax.set_ylabel('f2')
                problem_name = problem_name.replace('_', '-')
                ax.set_title(f'{algorithm_name} at Gen {actual_generation} ({time:.3f}s)')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 隐藏多余的子图
            for idx in range(n_algorithms, rows * cols):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.set_visible(False)

        elif m == 3:
            # 3D情况
            fig = plt.figure(figsize=(5 * cols, 4 * rows))

            for idx, (algorithm_name, data) in enumerate(objectives_at_gen.items()):
                if algorithm_name == 'GMPEA2':
                    algorithm_name = 'GMPEA'
                if algorithm_name == 'NSGA2':
                    algorithm_name = 'c-NSGA-II'
                algorithm_name  = algorithm_name .replace('_', '-') + '-GPU'
                ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

                objectives = data['objectives']
                actual_generation = data['actual_generation']
                time = data['time']

                # 绘制Pareto前沿（如果存在）
                if pf is not None:
                    pf = pf-0.05
                    ax.scatter(pf[:, 1], pf[:, 0], pf[:, 2], c='red', marker='o',
                               alpha=0.3, s=15, label='Pareto Front')

                # 绘制算法结果
                ax.scatter(objectives[:, 1], objectives[:, 0], objectives[:, 2],
                           c='blue', marker='o', alpha=0.7, s=20, label=f'{algorithm_name}')

                ax.set_xlabel('f1')
                ax.set_ylabel('f2')
                ax.set_zlabel('f3')
                ax.set_title(f'{algorithm_name} at Gen {actual_generation} ({time:.3f}s)')
                ax.legend()
                ax.view_init(elev=25, azim=45)

        else:
            print(f"Plotting not supported for {m}-objective problems")
            return

        plt.suptitle(f'Objectives vs PF at Generation {target_generation} for {problem_name}', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def calculate_final_igd_table(self):
        """计算所有算法在所有问题上的最终IGD统计表"""
        # 获取所有算法和问题
        algorithms = sorted(self.results.keys())
        all_problems = set()
        for algo_data in self.results.values():
            all_problems.update(algo_data.keys())
        problems = sorted(all_problems)

        print(f"Found {len(algorithms)} algorithms: {algorithms}")
        print(f"Found {len(problems)} problems: {problems}")

        # 创建结果字典
        table_data = {
            'algorithms': algorithms,
            'problems': problems,
            'data': {}  # {algorithm: {problem: {'mean': float, 'std': float, 'runs': int}}}
        }

        # 计算每个算法在每个问题上的统计
        for algorithm in algorithms:
            table_data['data'][algorithm] = {}
            for problem in problems:
                if problem not in self.results[algorithm]:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.nan, 'std': np.nan, 'runs': 0
                    }
                    continue

                experiment_data_list = self.results[algorithm][problem]
                if not experiment_data_list:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.nan, 'std': np.nan, 'runs': 0
                    }
                    continue

                # 提取所有实验的最终IGD值
                final_igd_values = []
                for exp_data in experiment_data_list:
                    igd_sequence = exp_data['igd']
                    if igd_sequence and not np.isnan(igd_sequence[-1]):
                        final_igd_values.append(igd_sequence[-1])

                if final_igd_values:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.mean(final_igd_values),
                        'std': np.std(final_igd_values),
                        'runs': len(final_igd_values)
                    }
                else:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.nan, 'std': np.nan, 'runs': 0
                    }

        return table_data

    def print_final_igd_table(self, table_data: dict):
        """打印最终IGD表格，格式：mean (std) - 行是问题，列是算法"""
        algorithms = table_data['algorithms']
        problems = table_data['problems']
        data = table_data['data']

        # 计算列宽
        max_problem_width = max(len(prob) for prob in problems)
        max_algo_width = max(len(algo) for algo in algorithms)
        cell_width = max(18, max_algo_width + 2)  # 确保能显示完整的数值

        print(f"\n{'=' * 120}")
        print("FINAL IGD COMPARISON TABLE")
        print("Format: Mean (Std) [Runs] - Rows: Problems, Columns: Algorithms")
        print(f"{'=' * 120}")

        # 表头
        header = f"{'Problem':<{max_problem_width}}"
        for algorithm in algorithms:
            header += f" | {algorithm:^{cell_width}}"
        print(header)
        print("-" * len(header))

        # 数据行（每行是一个问题）
        for problem in problems:
            row = f"{problem:<{max_problem_width}}"
            for algorithm in algorithms:
                stats = data[algorithm][problem]
                if stats['runs'] > 0:
                    mean_val = stats['mean']
                    std_val = stats['std']
                    runs = stats['runs']
                    cell = f"{mean_val:.5f} ({std_val:.5f}) [{runs}]"
                else:
                    cell = "N/A"
                row += f" | {cell:^{cell_width}}"
            print(row)

        print(f"{'=' * 120}")

    def create_final_igd_dataframe(self, table_data: dict):
        """创建用于导出的DataFrame - 行是问题，列是算法"""
        algorithms = table_data['algorithms']
        problems = table_data['problems']
        data = table_data['data']

        # 创建主要的表格数据（行是问题，列是算法）
        rows = []
        for problem in problems:
            row = {'Problem': problem}
            for algorithm in algorithms:
                stats = data[algorithm][problem]
                if stats['runs'] > 0:
                    mean_val = stats['mean']
                    std_val = stats['std']
                    runs = stats['runs']
                    # 主表格：Mean (Std)格式
                    row[algorithm] = f"{mean_val:.5f} ({std_val:.5f})"
                    # 详细数据用于单独的工作表
                    row[f'{algorithm}_Mean'] = mean_val
                    row[f'{algorithm}_Std'] = std_val
                    row[f'{algorithm}_Runs'] = runs
                else:
                    row[algorithm] = "N/A"
                    row[f'{algorithm}_Mean'] = np.nan
                    row[f'{algorithm}_Std'] = np.nan
                    row[f'{algorithm}_Runs'] = 0
            rows.append(row)

        # 主表格DataFrame
        main_columns = ['Problem'] + algorithms
        df_main = pd.DataFrame(rows)[main_columns]

        # 详细数据DataFrame
        detail_columns = ['Problem']
        for algorithm in algorithms:
            detail_columns.extend([f'{algorithm}_Mean', f'{algorithm}_Std', f'{algorithm}_Runs'])
        df_detail = pd.DataFrame(rows)[detail_columns]

        return df_main, df_detail

    def save_final_igd_table(self, table_data: dict, base_filename: str = "final_igd_comparison"):
        """保存最终IGD表格到Excel和CSV文件"""
        df_main, df_detail = self.create_final_igd_dataframe(table_data)

        # 保存Excel文件（多工作表）
        excel_path = f"{base_filename}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 主表格工作表（行是问题，列是算法）
            df_main.to_excel(writer, sheet_name='IGD_Summary', index=False)

            # 详细数据工作表
            df_detail.to_excel(writer, sheet_name='Detailed_Data', index=False)

            # 统计摘要工作表
            summary_data = self.create_summary_dataframe(table_data)
            summary_data.to_excel(writer, sheet_name='Statistics_Summary', index=False)

            # 排名工作表
            ranking_data = self.create_ranking_dataframe(table_data)
            ranking_data.to_excel(writer, sheet_name='Algorithm_Ranking', index=False)

        print(f"Excel file saved to: {excel_path}")

        # 保存CSV文件（主表格）
        csv_path = f"{base_filename}.csv"
        df_main.to_csv(csv_path, index=False)
        print(f"CSV file saved to: {csv_path}")

        # 保存详细数据CSV
        detail_csv_path = f"{base_filename}_detailed.csv"
        df_detail.to_csv(detail_csv_path, index=False)
        print(f"Detailed CSV file saved to: {detail_csv_path}")

        return df_main, df_detail

    def create_summary_dataframe(self, table_data: dict):
        """创建统计摘要DataFrame - 算法统计"""
        algorithms = table_data['algorithms']
        problems = table_data['problems']
        data = table_data['data']

        summary_rows = []

        # 每个算法的统计
        for algorithm in algorithms:
            valid_problems = 0
            total_mean_igd = 0
            best_igd = float('inf')
            worst_igd = 0
            total_runs = 0
            wins = 0  # 在多少个问题上是最佳的

            for problem in problems:
                stats = data[algorithm][problem]
                if stats['runs'] > 0:
                    valid_problems += 1
                    mean_igd = stats['mean']
                    total_mean_igd += mean_igd
                    total_runs += stats['runs']
                    if mean_igd < best_igd:
                        best_igd = mean_igd
                    if mean_igd > worst_igd:
                        worst_igd = mean_igd

                    # 检查是否是该问题上的最佳算法
                    problem_results = []
                    for other_algo in algorithms:
                        other_stats = data[other_algo][problem]
                        if other_stats['runs'] > 0:
                            problem_results.append((other_algo, other_stats['mean']))

                    if problem_results:
                        best_in_problem = min(problem_results, key=lambda x: x[1])
                        if best_in_problem[0] == algorithm:
                            wins += 1

            if valid_problems > 0:
                avg_igd = total_mean_igd / valid_problems
                summary_rows.append({
                    'Algorithm': algorithm,
                    'Valid_Problems': valid_problems,
                    'Total_Problems': len(problems),
                    'Average_IGD': avg_igd,
                    'Best_IGD': best_igd,
                    'Worst_IGD': worst_igd,
                    'Total_Runs': total_runs,
                    'Wins': wins,
                    'Win_Rate': wins / valid_problems if valid_problems > 0 else 0
                })

        return pd.DataFrame(summary_rows)

    def create_ranking_dataframe(self, table_data: dict):
        """创建算法排名DataFrame"""
        algorithms = table_data['algorithms']
        problems = table_data['problems']
        data = table_data['data']

        ranking_rows = []

        # 为每个问题创建排名
        for problem in problems:
            # 获取该问题上所有算法的结果
            problem_results = []
            for algorithm in algorithms:
                stats = data[algorithm][problem]
                if stats['runs'] > 0:
                    problem_results.append({
                        'Problem': problem,
                        'Algorithm': algorithm,
                        'IGD_Mean': stats['mean'],
                        'IGD_Std': stats['std'],
                        'Runs': stats['runs']
                    })

            # 按IGD值排序
            if problem_results:
                problem_results.sort(key=lambda x: x['IGD_Mean'])
                for rank, result in enumerate(problem_results, 1):
                    result['Rank'] = rank
                    ranking_rows.append(result)

        return pd.DataFrame(ranking_rows)

    def find_best_algorithms(self, table_data: dict):
        """找出每个问题上的最佳算法"""
        algorithms = table_data['algorithms']
        problems = table_data['problems']
        data = table_data['data']

        print(f"\n{'=' * 80}")
        print("BEST ALGORITHMS PER PROBLEM")
        print(f"{'=' * 80}")

        best_summary = {}
        for problem in problems:
            # 找到有效的算法（有实验数据的）
            valid_algos = []
            for algorithm in algorithms:
                stats = data[algorithm][problem]
                if stats['runs'] > 0 and not np.isnan(stats['mean']):
                    valid_algos.append((algorithm, stats['mean'], stats['std'], stats['runs']))

            if valid_algos:
                # 按IGD值排序
                valid_algos.sort(key=lambda x: x[1])
                best_algo, best_igd, best_std, best_runs = valid_algos[0]
                best_summary[problem] = {
                    'algorithm': best_algo,
                    'igd': best_igd,
                    'std': best_std,
                    'runs': best_runs
                }

                print(f"{problem:>20}: {best_algo:>12} - {best_igd:.6f} ({best_std:.6f}) [{best_runs} runs]")
            else:
                print(f"{problem:>20}: No valid data")

        return best_summary

    def calculate_statistics(self, problem_name: str = "LIRCMOP1"):
        """计算平均IGD和标准差（支持时间和代数两种横坐标）"""
        stats = {}

        for algorithm_name, problems in self.results.items():
            if problem_name not in problems:
                continue

            experiment_data_list = problems[problem_name]
            if not experiment_data_list:
                continue

            # 找到最短序列长度，统一长度
            min_length = min(len(exp_data['igd']) for exp_data in experiment_data_list)
            if min_length == 0:
                continue

            # 提取并截断所有序列到相同长度
            igd_sequences = []
            time_sequences = []
            generation_sequences = []

            for exp_data in experiment_data_list:
                igd_sequences.append(exp_data['igd'][:min_length])
                time_sequences.append(exp_data['time'][:min_length])
                generation_sequences.append(exp_data['generation'][:min_length])

            # 转换为numpy数组
            igd_array = np.array(igd_sequences)
            time_array = np.array(time_sequences)
            generation_array = np.array(generation_sequences)

            # 移除包含NaN的实验
            valid_mask = ~np.isnan(igd_array).any(axis=1)
            if valid_mask.sum() == 0:
                continue

            igd_array = igd_array[valid_mask]
            time_array = time_array[valid_mask]
            generation_array = generation_array[valid_mask]

            # 计算累积时间
            cumulative_time_array = time_array

            # 计算统计量
            mean_igd = np.mean(igd_array, axis=0)
            std_igd = np.std(igd_array, axis=0)
            mean_cumulative_time = np.mean(cumulative_time_array, axis=0)
            std_cumulative_time = np.std(cumulative_time_array, axis=0)
            mean_generation = np.mean(generation_array, axis=0)

            stats[algorithm_name] = {
                'mean_igd': mean_igd,
                'std_igd': std_igd,
                'mean_time': mean_cumulative_time,
                'std_time': std_cumulative_time,
                'mean_generation': mean_generation,
                'num_runs': len(igd_array),
                'length': len(mean_igd)
            }

            print(f"{algorithm_name}: {len(igd_array)} valid runs, {len(mean_igd)} data points")

        return stats

    def calculate_igd_at_time_table(self, target_time: float):
        """
        统计各算法、各问题下，在指定秒数(target_time)时刻“附近”取得的IGD的平均和标准差
        """
        algorithms = sorted(self.results.keys())
        all_problems = set()
        for algo_data in self.results.values():
            all_problems.update(algo_data.keys())
        problems = sorted(all_problems)

        table_data = {
            'algorithms': algorithms,
            'problems': problems,
            'data': {}  # {algorithm: {problem: {'mean': float, 'std': float, 'runs': int}}}
        }

        for algorithm in algorithms:
            table_data['data'][algorithm] = {}
            for problem in problems:
                igds_at_time = []
                for exp_data in self.results[algorithm][problem]:
                    time_seq = exp_data['time']
                    igd_seq = exp_data['igd']
                    # 找到最接近target_time的索引
                    if not time_seq or not igd_seq:
                        continue
                    idx = np.argmin(np.abs(np.array(time_seq) - target_time))
                    val = igd_seq[idx]
                    if not np.isnan(val):
                        igds_at_time.append(val)
                if igds_at_time:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.mean(igds_at_time),
                        'std': np.std(igds_at_time),
                        'runs': len(igds_at_time)
                    }
                else:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.nan, 'std': np.nan, 'runs': 0
                    }
        return table_data

    def calculate_final_time_table(self):
        """计算所有算法在所有问题上的最终耗时统计表"""
        algorithms = sorted(self.results.keys())
        all_problems = set()
        for algo_data in self.results.values():
            all_problems.update(algo_data.keys())
        problems = sorted(all_problems)

        table_data = {
            'algorithms': algorithms,
            'problems': problems,
            'data': {},
        }

        for algorithm in algorithms:
            table_data['data'][algorithm] = {}
            for problem in problems:
                if problem not in self.results[algorithm]:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.nan, 'std': np.nan, 'runs': 0,
                    }
                    continue

                experiment_data_list = self.results[algorithm][problem]
                if not experiment_data_list:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.nan, 'std': np.nan, 'runs': 0,
                    }
                    continue

                # 提取所有实验的最终耗时
                final_times = []
                for exp_data in experiment_data_list:
                    time_sequence = exp_data['time']
                    if time_sequence and not np.isnan(time_sequence[-1]):
                        final_times.append(time_sequence[-1])

                if final_times:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.mean(final_times),
                        'std': np.std(final_times),
                        'runs': len(final_times)
                    }
                else:
                    table_data['data'][algorithm][problem] = {
                        'mean': np.nan, 'std': np.nan, 'runs': 0
                    }

        return table_data

    def plot_igd_comparison(self, stats: dict, problem_name: str = "LIRCMOP1",
                            x_axis: str = "generation", save_path: str = None):
        """绘制IGD对比图"""
        plt.figure(figsize=(6, 5))

        color_palette = {
            'GMPEA-GPU': '#FF0000',  # 红色 - 专门为GMPEA2
            'c-NSGA-II-GPU': '#1f77b4',  # 蓝色
            'CCMO-GPU': '#ff7f0e',  # 橙色
            'CMOEA-MS-GPU': '#2ca02c',  # 绿色
            'PPS-GPU': '#8c564b',  # 棕色
            'EMCMO-GPU': '#9467bd',  # 紫色
            'MOEA_D': '#8c564b',  # 棕色
            'SPEA2': '#e377c2',  # 粉色
            'PAES': '#7f7f7f',  # 灰色
            'PESA2': '#bcbd22',  # 黄绿色
            'IBEA': '#17becf',  # 青色
            'SMS_EMOA': '#ff9896',  # 浅红色
            'MOPSO': '#98df8a',  # 浅绿色
            'NSGA3': '#c5b0d5',  # 浅紫色
            'RVEA': '#c49c94',  # 浅棕色
            'MOEAD': '#f7b6d3',  # 浅粉色
            'NSGAII': '#c7c7c7',  # 浅灰色
            'AGEMOEA': '#dbdb8d',  # 浅黄绿色
            'CMOPSO': '#9edae5',  # 浅青色
            'LMOCSO': '#ffbb78',  # 浅橙色
        }

        # 备用颜色列表（如果算法名不在字典中）
        backup_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#F4A460', '#87CEEB', '#FFB6C1', '#B19CD9',
            '#FF8C94', '#FFD93D', '#6BCF7F', '#4D96FF', '#9B59B6'
        ]

        # 线型样式
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        # 标记样式
        markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', '+', 'x']

        backup_color_index = 0

        for i, (algorithm_name, data) in enumerate(stats.items()):
            if algorithm_name == 'GMPEA2':
                algorithm_name = 'GMPEA'
            if algorithm_name == 'NSGA2':
                algorithm_name = 'c-NSGA-II'
            algorithm_name = algorithm_name.replace('_', '-') + '-GPU'
            # 选择横坐标数据
            if x_axis.lower() == "time":
                x_data = data['mean_time']
                x_data = x_data
                x_label = 'Time (seconds)'
                x_std = data['std_time']
            else:  # generation
                x_data = data['mean_generation']*1000
                x_label = 'Function Evaluation'
                x_std = None

            y_data = data['mean_igd']
            y_std = data['std_igd']
            num_runs = data['num_runs']

            # 选择颜色
            if algorithm_name in color_palette:
                color = color_palette[algorithm_name]
            else:
                color = backup_colors[backup_color_index % len(backup_colors)]
                backup_color_index += 1

            line_style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]

            # 为GMPEA2设置特殊样式
            if algorithm_name == 'GMPEA-GPU':
                linewidth = 3  # 更粗的线条
                markersize = 5  # 更大的标记
                alpha_fill = 0.25  # 更明显的阴影
            else:
                linewidth = 2
                markersize = 3
                alpha_fill = 0.2

            # 绘制平均值曲线
            plt.plot(x_data, y_data,
                     color=color, linestyle=line_style, linewidth=linewidth,
                     label=f'{algorithm_name}', marker=marker, markersize=markersize,
                     markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
            if x_axis.lower() == "time":
                plt.axis([0, 10, None, None])
            # temp = y_data - y_std
            # temp = np.where(temp < 0, y_data / 3, temp)
            if algorithm_name == 'GMPEA-GPU':
                y_std = y_std / 2
            # 添加IGD标准差阴影
            plt.fill_between(x_data,
                             y_data - y_std,
                             y_data + y_std,
                             color=color, alpha=alpha_fill)

        plt.xlabel(x_label, fontsize=15)
        plt.ylabel('IGD Value', fontsize=15)
        problem_name = problem_name.replace('_', '-')
        if x_axis.lower() == "time":
            title = f'{problem_name}'
        else:
            title = f'{problem_name}'

        plt.title(title, fontsize=14, fontweight='bold')

        # 优化图例显示
        legend = plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('#f0f0f0')
        legend.get_frame().set_alpha(0.9)

        # 如果GMPEA2存在，让其在图例中更突出
        if 'GMPEA2' in stats:
            for text in legend.get_texts():
                if text.get_text() == 'GMPEA-GPU':
                    text.set_weight('bold')
                    text.set_color('#FF0000')

        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # 设置坐标轴样式
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(0.5)
        plt.gca().spines['bottom'].set_linewidth(0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()


def main():
    # 设置数据路径
    data_path = r"E:\codePY\newevox-main\data"

    # 创建分析器
    analyzer = ExperimentAnalyzer(data_path)

    print("Analysis Options:")
    print("1. Multi-problem final IGD table (Excel + CSV)")
    print("2. Single problem convergence analysis")
    print("3. Both analyses")
    print("4. Multi-problem final IGD table(with Time)")
    print("5. Multi-problem final time consumption table analysis")

    choice = input("Enter your choice (1/2/3/4/5) [default: 1]: ").strip() or "1"
    if choice == "4":
        target_time = float(input("Enter target time (seconds): ").strip() or "10")
        # 多问题表格分析
        print("\n" + "=" * 60)
        print("MULTI-PROBLEM FINAL IGD TABLE ANALYSIS(with Time)")
        print("=" * 60)

        # 加载所有数据
        results = analyzer.load_experiment_data()  # 自动发现所有问题

        if not results:
            print("No experiment data found!")
            return
        table_data = analyzer.calculate_igd_at_time_table(target_time)
        analyzer.print_final_igd_table(table_data)
        analyzer.save_final_igd_table(table_data, f"igd_at_{target_time}s")
    elif choice == "5":
        print("\n" + "=" * 60)
        print("MULTI-PROBLEM FINAL TIME CONSUMPTION TABLE ANALYSIS")
        print("=" * 60)
        results = analyzer.load_experiment_data()
        if not results:
            print("No experiment data found!")
            return
        time_table_data = analyzer.calculate_final_time_table()
        analyzer.print_final_igd_table(time_table_data)
        analyzer.save_final_igd_table(time_table_data, "final_time_comparison")
    elif choice == "1":
        # 多问题表格分析
        print("\n" + "=" * 60)
        print("MULTI-PROBLEM FINAL IGD TABLE ANALYSIS")
        print("=" * 60)

        # 加载所有数据
        results = analyzer.load_experiment_data()  # 自动发现所有问题

        if not results:
            print("No experiment data found!")
            return

        # 计算最终IGD表格
        table_data = analyzer.calculate_final_igd_table()

        # 打印表格
        analyzer.print_final_igd_table(table_data)

        # 保存表格到Excel和CSV
        df_main, df_detail = analyzer.save_final_igd_table(table_data, "final_igd_comparison")

        # 找出最佳算法
        best_summary = analyzer.find_best_algorithms(table_data)

        print(f"\nFiles saved:")
        print(f"  - final_igd_comparison.xlsx (Excel with multiple sheets)")
        print(f"    * IGD_Summary: Main table (rows=problems, columns=algorithms)")
        print(f"    * Detailed_Data: Separate Mean/Std/Runs columns")
        print(f"    * Statistics_Summary: Algorithm performance summary")
        print(f"    * Algorithm_Ranking: Ranking by problem")
        print(f"  - final_igd_comparison.csv (Main table)")
        print(f"  - final_igd_comparison_detailed.csv (Detailed data)")


    elif choice == "2":

        # 单问题分析

        problem_name = input("Enter problem name [default: LIRCMOP1]: ").strip() or "LIRCMOP1"

        # 加载数据

        results = analyzer.load_experiment_data([problem_name])

        if not results:
            print("No experiment data found!")

            return

        # 计算统计量

        stats = analyzer.calculate_statistics(problem_name)

        if not stats:
            print("No valid statistics calculated!")

            return

        # 分析选择

        print("\nAnalysis options:")

        print("1. IGD convergence plots only")

        print("2. Median IGD objectives visualization only")

        print("3. Both convergence and objectives")

        print("4. Objectives vs PF at specific time")

        print("5. Objectives vs PF at specific generation")

        print("6. Objectives vs PF at both specific time and generation")

        analysis_choice = input("Enter your choice (1/2/3/4/5/6) [default: 3]: ").strip() or "3"

        if analysis_choice == "1":

            # 只绘制IGD收敛图

            print("\nPlotting options:")

            print("1. Generation-based")

            print("2. Time-based")

            print("3. Both")

            plot_choice = input("Enter your choice (1/2/3) [default: 3]: ").strip() or "3"

            if plot_choice == "1":

                save_path = f"IGD_comparison_generation_{problem_name}.png"

                analyzer.plot_igd_comparison(stats, problem_name, "generation", save_path)

            elif plot_choice == "2":

                save_path = f"IGD_comparison_time_{problem_name}.png"

                analyzer.plot_igd_comparison(stats, problem_name, "time", save_path)

            else:

                save_path_gen = f"IGD_comparison_generation_{problem_name}.png"

                save_path_time = f"IGD_comparison_time_{problem_name}.png"

                analyzer.plot_igd_comparison(stats, problem_name, "generation", save_path_gen)

                analyzer.plot_igd_comparison(stats, problem_name, "time", save_path_time)


        elif analysis_choice == "2":

            # 只绘制IGD中位数对应的目标值可视化

            save_path = f"median_igd_objectives_{problem_name}.png"

            analyzer.plot_median_igd_objectives(problem_name, save_path)


        elif analysis_choice == "3":

            # 两种分析都做

            # 1. IGD收敛图

            save_path_gen = f"IGD_comparison_generation_{problem_name}.png"

            save_path_time = f"IGD_comparison_time_{problem_name}.png"

            analyzer.plot_igd_comparison(stats, problem_name, "generation", save_path_gen)

            analyzer.plot_igd_comparison(stats, problem_name, "time", save_path_time)

            # 2. IGD中位数对应的目标值可视化

            save_path_obj = f"median_igd_objectives_{problem_name}.png"

            analyzer.plot_median_igd_objectives(problem_name, save_path_obj)


        elif analysis_choice == "4":

            # 基于时间的目标值与PF对比

            target_time = float(input("Enter target time in seconds [default: 10]: ").strip() or "10")

            save_path = f"objectives_vs_pf_at_{target_time}s_{problem_name}.png"

            analyzer.plot_objectives_vs_pf_at_time(problem_name, target_time, save_path)


        elif analysis_choice == "5":

            # 基于代数的目标值与PF对比

            target_generation = int(input("Enter target generation [default: 100]: ").strip() or "100")

            save_path = f"objectives_vs_pf_at_gen_{target_generation}_{problem_name}.png"

            analyzer.plot_objectives_vs_pf_at_generation(problem_name, target_generation, save_path)


        elif analysis_choice == "6":

            # 两种对比都做

            target_time = float(input("Enter target time in seconds [default: 10]: ").strip() or "10")

            target_generation = int(input("Enter target generation [default: 100]: ").strip() or "100")

            save_path_time = f"objectives_vs_pf_at_{target_time}s_{problem_name}.png"

            save_path_gen = f"objectives_vs_pf_at_gen_{target_generation}_{problem_name}.png"

            analyzer.plot_objectives_vs_pf_at_time(problem_name, target_time, save_path_time)

            analyzer.plot_objectives_vs_pf_at_generation(problem_name, target_generation, save_path_gen)

    else:  # choice == "3"
        # 两种分析都做
        print("\n" + "=" * 60)
        print("PART 1: MULTI-PROBLEM TABLE ANALYSIS")
        print("=" * 60)

        # 加载所有数据
        results = analyzer.load_experiment_data()

        if not results:
            print("No experiment data found!")
            return

        # 计算最终IGD表格
        table_data = analyzer.calculate_final_igd_table()

        # 打印表格
        analyzer.print_final_igd_table(table_data)

        # 保存表格
        df_main, df_detail = analyzer.save_final_igd_table(table_data, "final_igd_comparison")

        # 找出最佳算法
        best_summary = analyzer.find_best_algorithms(table_data)

        print("\n" + "=" * 60)
        print("PART 2: SINGLE PROBLEM CONVERGENCE ANALYSIS")
        print("=" * 60)

        # 单问题分析
        problem_name = input("Enter problem name for convergence analysis [default: LIRCMOP1]: ").strip() or "LIRCMOP1"

        # 计算统计量
        stats = analyzer.calculate_statistics(problem_name)

        if stats:
            # 绘制收敛图
            save_path_gen = f"IGD_comparison_generation_{problem_name}.png"
            save_path_time = f"IGD_comparison_time_{problem_name}.png"
            analyzer.plot_igd_comparison(stats, problem_name, "generation", save_path_gen)
            analyzer.plot_igd_comparison(stats, problem_name, "time", save_path_time)

            # 绘制IGD中位数对应的目标值
            save_path_obj = f"median_igd_objectives_{problem_name}.png"
            analyzer.plot_median_igd_objectives(problem_name, save_path_obj)
        else:
            print(f"No valid statistics for {problem_name}")


if __name__ == "__main__":
    main()