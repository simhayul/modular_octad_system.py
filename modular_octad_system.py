"""
Modular Octad System: 메타 프롬프트 재귀적 선순환 기반 AI 거버넌스
핵심 혁신:
1. 5×5 모듈 구조 (L1~L5 난이도별 독립 채널)
2. 메타 프롬프트 진화 (프롬프트 생성 지침 자체를 진화)
3. Persistent Bonus with Decay (인플레이션 방지)
4. Hidden Test Cases (일반화 능력 검증)
5. 재귀적 선순환 피드백 루프

개선사항:
- Bonus Decay로 Goodhart's Law 대응
- 하드코딩 감지 및 패널티
- 결과 저장 기능 (CSV, JSON)
- 3회 반복 실험 지원
"""

import os
import json
import time
import statistics
import random
import re
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed!")
    print("Install with: pip install openai")
    exit(1)


@dataclass
class MetaPrompt:
    """메타 프롬프트: 프롬프트를 생성하는 지침"""
    id: str
    difficulty_level: int
    rules: List[str] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    is_elite: bool = False

    @property
    def average_fitness(self) -> float:
        return statistics.mean(self.fitness_history) if self.fitness_history else 0.0

    def to_instruction(self) -> str:
        """메타 프롬프트를 실제 지침 텍스트로 변환"""
        instruction = f"[Level {self.difficulty_level} Problem Solving Guidelines]\n\n"
        instruction += "RULES:\n"
        for i, rule in enumerate(self.rules, 1):
            instruction += f"{i}. {rule}\n"
        instruction += "\nProblem: {problem}\n\n"
        instruction += "Generate a Python function that solves the above problem following these rules."
        return instruction


@dataclass
class Prompt:
    """실제 프롬프트"""
    id: str
    meta_prompt_id: str
    content: str
    difficulty_level: int
    fitness: float = 0.0


@dataclass
class Problem:
    """문제 정의"""
    id: str
    description: str
    test_cases: List[Dict] = field(default_factory=list)
    hidden_test_cases: List[Dict] = field(default_factory=list)  # 추가!
    difficulty_level: int = 1
    base_weight: float = 1.0


@dataclass
class Solution:
    """생성된 솔루션"""
    content: str
    prompt_id: str
    problem_id: str
    correctness: float = 0.0
    quality: float = 0.0
    fitness: float = 0.0


class LAlphaEngine:
    """난이도별 메타 프롬프트 진화 엔진"""

    def __init__(self, difficulty_level: int, initial_rules: List[str]):
        self.difficulty_level = difficulty_level
        self.meta_prompts: Dict[str, MetaPrompt] = {}
        self.generation = 0

        self._initialize_meta_prompts(initial_rules)
        self.elite_ratio = self._get_elite_ratio()
        self.mutation_rate = 0.4
        self.crossover_rate = 0.3

    def _get_elite_ratio(self) -> float:
        """난이도별 엘리트 보존 비율"""
        ratios = {1: 0.5, 2: 0.4, 3: 0.35, 4: 0.3, 5: 0.25}
        return ratios.get(self.difficulty_level, 0.3)

    def _initialize_meta_prompts(self, initial_rules: List[str]):
        """초기 메타 프롬프트 생성"""
        for i in range(4):
            mp_id = f"L{self.difficulty_level}_mp0_{i}"
            self.meta_prompts[mp_id] = MetaPrompt(
                id=mp_id,
                difficulty_level=self.difficulty_level,
                rules=initial_rules.copy(),
                generation=0,
                is_elite=True
            )

    def generate_prompt(self, meta_prompt_id: str) -> Prompt:
        """메타 프롬프트로부터 실제 프롬프트 생성"""
        meta_prompt = self.meta_prompts[meta_prompt_id]
        instruction = meta_prompt.to_instruction()

        prompt_id = f"{meta_prompt_id}_p{int(time.time()*1000)}"
        return Prompt(
            id=prompt_id,
            meta_prompt_id=meta_prompt_id,
            content=instruction,
            difficulty_level=self.difficulty_level
        )

    def evolve(self, forced_exploration: bool = False):
        """메타 프롬프트 진화"""
        self.generation += 1

        sorted_mps = sorted(
            self.meta_prompts.values(),
            key=lambda x: x.average_fitness,
            reverse=True
        )

        num_elites = max(2, int(len(sorted_mps) * self.elite_ratio))
        elites = sorted_mps[:num_elites]

        for mp in self.meta_prompts.values():
            mp.is_elite = False
        for elite in elites:
            elite.is_elite = True

        new_mps = []

        if random.random() < self.crossover_rate and len(elites) >= 2:
            parent1, parent2 = random.sample(elites, 2)
            child = self._crossover(parent1, parent2)
            if child:
                new_mps.append(child)

        if random.random() < self.mutation_rate:
            parent = random.choice(elites)
            mutant = self._mutate(parent, radical=forced_exploration)
            if mutant:
                new_mps.append(mutant)

        for mp in new_mps:
            self.meta_prompts[mp.id] = mp

        if len(self.meta_prompts) > 10:
            self._prune_weak()

        print(f"  [L{self.difficulty_level}-Alpha] Evolved: {len(new_mps)} new, "
              f"{num_elites} elites, {len(self.meta_prompts)} total")

    def _crossover(self, mp1: MetaPrompt, mp2: MetaPrompt) -> Optional[MetaPrompt]:
        """교차 결합"""
        mid1 = len(mp1.rules) // 2
        mid2 = len(mp2.rules) // 2

        new_rules = mp1.rules[:mid1] + mp2.rules[mid2:]

        seen = set()
        unique_rules = []
        for rule in new_rules:
            if rule not in seen:
                seen.add(rule)
                unique_rules.append(rule)

        mp_id = f"L{self.difficulty_level}_mp{self.generation}_c{random.randint(1000,9999)}"
        return MetaPrompt(
            id=mp_id,
            difficulty_level=self.difficulty_level,
            rules=unique_rules,
            generation=self.generation,
            parent_ids=[mp1.id, mp2.id]
        )

    def _mutate(self, parent: MetaPrompt, radical: bool = False) -> Optional[MetaPrompt]:
        """변이"""
        new_rules = parent.rules.copy()

        standard_mutations = [
            "Always include docstrings",
            "Handle edge cases explicitly",
            "Optimize for time complexity",
            "Use descriptive variable names",
            "Add inline comments for complex logic"
        ]

        radical_mutations = [
            "Try multiple algorithmic approaches",
            "Use advanced data structures",
            "Implement memoization for recursion",
            "Consider trade-offs between readability and performance"
        ]

        mutation_pool = radical_mutations if radical else standard_mutations
        new_rule = random.choice(mutation_pool)

        if new_rule not in new_rules:
            new_rules.append(new_rule)

        mp_id = f"L{self.difficulty_level}_mp{self.generation}_m{random.randint(1000,9999)}"
        return MetaPrompt(
            id=mp_id,
            difficulty_level=self.difficulty_level,
            rules=new_rules,
            generation=self.generation,
            parent_ids=[parent.id]
        )

    def _prune_weak(self):
        """약한 메타 프롬프트 제거"""
        sorted_mps = sorted(
            self.meta_prompts.values(),
            key=lambda x: x.average_fitness,
            reverse=True
        )
        keep = sorted_mps[:10]
        self.meta_prompts = {mp.id: mp for mp in keep}

    def learn_from_success(self, meta_prompt_id: str, successful_features: List[str]):
        """재귀적 선순환"""
        if meta_prompt_id not in self.meta_prompts:
            return

        mp = self.meta_prompts[meta_prompt_id]
        for feature in successful_features:
            if feature not in mp.rules:
                mp.rules.append(feature)
                print(f"  [L{self.difficulty_level}-Alpha] Learned: {feature}")


class LOmegaEngine:
    """난이도별 평가 엔진"""

    def __init__(self, difficulty_level: int, problem_pool: List[Problem]):
        self.difficulty_level = difficulty_level
        self.problem_pool = [p for p in problem_pool if p.difficulty_level == difficulty_level]
        self.base_weight = {1: 1.0, 2: 1.2, 3: 1.5, 4: 1.8, 5: 2.0}[difficulty_level]
        self.persistent_bonus = 0.0
        self.bonus_decay_rate = 0.95  # 5% 감쇠
        self.bonus_cap = 0.5  # 상한선
        self.evaluation_history: List[Dict] = []
        self.perfect_scores: List[str] = []

    def evaluate_solution(self, solution: Solution, problem: Problem) -> float:
        """솔루션 평가"""
        solution.correctness = self._evaluate_correctness(solution, problem)
        solution.quality = self._evaluate_quality(solution)

        base_fitness = 0.7 * solution.correctness + 0.3 * solution.quality
        weighted_fitness = base_fitness * self.base_weight

        if solution.correctness >= 0.99 and solution.quality >= 0.8:
            bonus_increment = 0.05 * self.base_weight  # 난이도별 차등
            self.persistent_bonus += bonus_increment
            self.perfect_scores.append(solution.prompt_id)
            print(f"  [L{self.difficulty_level}-Omega] Perfect! Bonus: +{self.persistent_bonus:.3f}")

        # 보너스 감쇠 및 상한 적용
        self.persistent_bonus *= self.bonus_decay_rate
        self.persistent_bonus = min(self.persistent_bonus, self.bonus_cap)

        final_fitness = weighted_fitness + self.persistent_bonus
        solution.fitness = final_fitness

        self.evaluation_history.append({
            'solution_id': solution.prompt_id,
            'problem_id': problem.id,
            'fitness': final_fitness,
            'correctness': solution.correctness,
            'quality': solution.quality,
            'timestamp': time.time()
        })

        return final_fitness

    def _evaluate_correctness(self, solution: Solution, problem: Problem) -> float:
        """Public과 Hidden 테스트 모두 평가"""
        all_tests = problem.test_cases + problem.hidden_test_cases
        if not all_tests:
            return 1.0

        passed = 0
        for test_case in all_tests:
            try:
                local_ns = {}
                exec(solution.content, {}, local_ns)
                func = next((v for v in local_ns.values() if callable(v)), None)
                if not func:
                    continue
                result = func(test_case.get('input')) if 'input' in test_case else func()
                if result == test_case['expected']:
                    passed += 1
            except:
                continue

        return passed / len(all_tests)

    def _evaluate_quality(self, solution: Solution) -> float:
        """품질 평가 with 하드코딩 패널티"""
        score = 0.5

        # 기존 점수
        if '"""' in solution.content or "'''" in solution.content:
            score += 0.15
        if '#' in solution.content:
            score += 0.1
        if 'def ' in solution.content:
            score += 0.15
        if 'return' in solution.content:
            score += 0.1

        # 하드코딩 감지 패널티
        hardcoded_patterns = [
            r"return\s+['\"]olleh['\"]",  # 'olleh' 같은 직접 답변
            r"return\s+['\"]dlrow['\"]",
            r"return\s+\d+\s*$",  # 단순 숫자 리턴
            r"if.*==.*['\"].*['\"].*return\s+['\"]",  # if input == X: return Y 패턴
        ]

        for pattern in hardcoded_patterns:
            if re.search(pattern, solution.content):
                score *= 0.5  # 50% 패널티
                print(f"  [L{self.difficulty_level}-Quality] Hardcoding detected! Penalty applied.")
                break

        return max(0.0, min(1.0, score))

    def analyze_success_patterns(self) -> List[str]:
        """성공 패턴 분석"""
        if len(self.evaluation_history) < 5:
            return []

        sorted_evals = sorted(self.evaluation_history, key=lambda x: x['fitness'], reverse=True)
        top_20_percent = sorted_evals[:max(1, len(sorted_evals)//5)]

        patterns = []
        if any(e['fitness'] > 1.5 for e in top_20_percent):
            patterns.append("Comprehensive error handling improves quality")
        if any(e['fitness'] > 2.0 for e in top_20_percent):
            patterns.append("Optimal algorithm selection is critical")

        return patterns


class DeltaEngine:
    """정체 감지 엔진"""

    def __init__(self):
        self.stagnation_threshold = 15
        self.stagnation_counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.last_best_fitness = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    def update(self, difficulty_level: int, current_best: float) -> bool:
        """정체 감지"""
        last_best = self.last_best_fitness[difficulty_level]

        if abs(current_best - last_best) < 0.02:
            self.stagnation_counters[difficulty_level] += 1
        else:
            self.stagnation_counters[difficulty_level] = 0

        self.last_best_fitness[difficulty_level] = current_best

        if self.stagnation_counters[difficulty_level] >= self.stagnation_threshold:
            print(f"  [Delta] L{difficulty_level} stagnation! Triggering exploration.")
            self.stagnation_counters[difficulty_level] = 0
            return True

        return False


class ChaosBarrier:
    """리소스 제한"""

    def __init__(self, max_iterations=50, max_llm_calls=200):
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.current_calls = 0

    def check_iteration(self, iteration: int) -> bool:
        return iteration >= self.max_iterations

    def check_llm_calls(self) -> bool:
        return self.current_calls >= self.max_llm_calls

    def register_call(self):
        self.current_calls += 1

    def reset(self):
        self.current_calls = 0


class ModularOctadFramework:
    """5×5 모듈 구조의 옥타드 시스템"""

    def __init__(self, problem_pool: List[Problem], generator_func: Callable):
        self.chaos_barrier = ChaosBarrier()
        self.delta = DeltaEngine()

        base_rules = [
            "Write clean, readable code",
            "Follow Python best practices",
            "Test with provided examples"
        ]

        self.alpha_engines = {}
        self.omega_engines = {}

        for L in range(1, 6):
            self.alpha_engines[L] = LAlphaEngine(L, base_rules)
            self.omega_engines[L] = LOmegaEngine(L, problem_pool)

        self.generator_func = generator_func
        self.iteration = 0
        self.results_log = []  # 결과 로깅

    def run(self, iterations: int = 30):
        """메인 실행 루프"""
        print("=" * 80)
        print("MODULAR OCTAD SYSTEM: 메타 프롬프트 재귀적 선순환 (개선 버전)")
        print("=" * 80)
        print(f"구조: 5×5 모듈 (L1~L5 독립 채널)")
        print(f"개선: Bonus Decay + Hidden Tests + Hardcoding Detection")
        print("=" * 80)

        for iteration in range(iterations):
            if self.chaos_barrier.check_iteration(iteration):
                break

            self.iteration = iteration
            self.chaos_barrier.reset()

            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"{'='*80}")

            for L in range(1, 6):
                self._run_level(L)

            if iteration > 0 and iteration % 5 == 0:
                print(f"\n[System] Evolution...")
                for L in range(1, 6):
                    forced = self.delta.stagnation_counters[L] > 10
                    self.alpha_engines[L].evolve(forced_exploration=forced)

        print("\n" + "=" * 80)
        print("EVOLUTION COMPLETE")
        print("=" * 80)
        self._print_summary()

    def _run_level(self, L: int):
        """특정 난이도 레벨 실행"""
        alpha = self.alpha_engines[L]
        omega = self.omega_engines[L]

        level_fitness = []

        for mp_id in list(alpha.meta_prompts.keys()):
            if self.chaos_barrier.check_llm_calls():
                break

            prompt = alpha.generate_prompt(mp_id)

            for problem in omega.problem_pool[:2]:
                self.chaos_barrier.register_call()
                solution = self._generate_solution(prompt, problem)
                fitness = omega.evaluate_solution(solution, problem)
                level_fitness.append(fitness)
                alpha.meta_prompts[mp_id].fitness_history.append(fitness)

        if level_fitness:
            avg_fit = statistics.mean(level_fitness)
            max_fit = max(level_fitness)

            # 로그 저장
            self.results_log.append({
                'iteration': self.iteration,
                'level': L,
                'avg_fitness': avg_fit,
                'max_fitness': max_fit,
                'bonus': omega.persistent_bonus,
                'num_meta_prompts': len(alpha.meta_prompts),
                'perfect_scores': len(omega.perfect_scores)
            })

            print(f"  [L{L}] Avg: {avg_fit:.3f} | Max: {max_fit:.3f} | "
                  f"Bonus: +{omega.persistent_bonus:.3f} | "
                  f"MPs: {len(alpha.meta_prompts)}")

            if self.delta.update(L, max_fit):
                patterns = omega.analyze_success_patterns()
                if patterns:
                    best_mp_id = max(alpha.meta_prompts.keys(),
                                     key=lambda x: alpha.meta_prompts[x].average_fitness)
                    alpha.learn_from_success(best_mp_id, patterns)

    def _generate_solution(self, prompt: Prompt, problem: Problem) -> Solution:
        """솔루션 생성"""
        full_prompt = prompt.content.replace("{problem}", problem.description)

        try:
            code = self.generator_func(full_prompt)
            return Solution(
                content=code,
                prompt_id=prompt.id,
                problem_id=problem.id
            )
        except Exception as e:
            print(f"  [Error] {e}")
            return Solution(
                content="def solution():\n    pass",
                prompt_id=prompt.id,
                problem_id=problem.id
            )

    def _print_summary(self):
        """최종 요약"""
        print("\nFINAL STATISTICS:")
        print("-" * 80)

        for L in range(1, 6):
            alpha = self.alpha_engines[L]
            omega = self.omega_engines[L]

            if alpha.meta_prompts:
                best_mp = max(alpha.meta_prompts.values(),
                              key=lambda x: x.average_fitness)
                avg_fitness = best_mp.average_fitness
            else:
                avg_fitness = 0.0

            print(f"Level {L}:")
            print(f"  Best Fitness: {avg_fitness:.3f}")
            print(f"  Persistent Bonus: +{omega.persistent_bonus:.3f}")
            print(f"  Perfect Scores: {len(omega.perfect_scores)}")
            print(f"  Meta Prompts: {len(alpha.meta_prompts)}")
            print()

    def save_results(self, filename_prefix="octad_results"):
        """결과를 파일로 저장"""
        # CSV로 iteration log 저장
        try:
            import pandas as pd
            df = pd.DataFrame(self.results_log)
            df.to_csv(f"{filename_prefix}_log.csv", index=False)
            print(f"✓ Log saved: {filename_prefix}_log.csv")
        except ImportError:
            # pandas 없으면 수동으로 CSV 작성
            with open(f"{filename_prefix}_log.csv", 'w') as f:
                if self.results_log:
                    headers = list(self.results_log[0].keys())
                    f.write(','.join(headers) + '\n')
                    for row in self.results_log:
                        f.write(','.join(str(row[h]) for h in headers) + '\n')
            print(f"✓ Log saved: {filename_prefix}_log.csv")

        # JSON으로 최종 결과 저장
        final_results = {
            'total_iterations': self.iteration + 1,
            'levels': {}
        }

        for L in range(1, 6):
            alpha = self.alpha_engines[L]
            omega = self.omega_engines[L]

            best_mp = max(alpha.meta_prompts.values(),
                          key=lambda x: x.average_fitness) if alpha.meta_prompts else None

            final_results['levels'][f'L{L}'] = {
                'best_fitness': best_mp.average_fitness if best_mp else 0.0,
                'persistent_bonus': omega.persistent_bonus,
                'perfect_scores': len(omega.perfect_scores),
                'num_meta_prompts': len(alpha.meta_prompts),
                'best_meta_prompt_rules': best_mp.rules if best_mp else []
            }

        with open(f"{filename_prefix}_final.json", 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"✓ Results saved: {filename_prefix}_final.json")


def create_openai_generator(model="gpt-4o-mini", temperature=0.7):
    """OpenAI 생성기"""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not set!")

    client = OpenAI(api_key=api_key)

    def generator(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )

            code = response.choices[0].message.content.strip()

            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

            return code
        except Exception as e:
            print(f"  [LLM Error] {e}")
            return "def solution():\n    pass"

    return generator


def run_multiple_experiments(problem_pool: List[Problem], num_runs: int = 3, iterations: int = 20):
    """통계적 유의성을 위한 반복 실험"""
    print("\n" + "=" * 80)
    print(f"RUNNING {num_runs} EXPERIMENTS FOR STATISTICAL SIGNIFICANCE")
    print("=" * 80 + "\n")

    for run in range(num_runs):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT RUN {run + 1}/{num_runs}")
        print(f"{'='*80}\n")

        generator = create_openai_generator()
        framework = ModularOctadFramework(problem_pool, generator)
        framework.run(iterations=iterations)
        framework.save_results(f"octad_run{run+1}")

        # 다음 실험 전 대기
        if run < num_runs - 1:
            print("\n⏳ Waiting 5 seconds before next run...")
            time.sleep(5)

    print("\n" + "=" * 80)
    print("✓ ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)


def main():
    """메인 함수"""
    print("=" * 80)
    print("MODULAR OCTAD SYSTEM (IMPROVED)")
    print("=" * 80)

    if not os.environ.get("API_KEY"):
        print("Error: API_KEY not set!")
        print("Set with: set API_KEY=your-key (Windows) or export API_KEY=your-key (Linux/Mac)")
        return

    # 문제 풀 정의 (Hidden Test Cases 포함)
    problem_pool = [
        Problem(
            id="reverse_string",
            description="Write 'reverse_string(s)' that returns reversed string.",
            difficulty_level=1,
            test_cases=[
                {'input': 'hello', 'expected': 'olleh'},
                {'input': 'world', 'expected': 'dlrow'}
            ],
            hidden_test_cases=[
                {'input': 'python', 'expected': 'nohtyp'},
                {'input': '', 'expected': ''},
                {'input': 'a', 'expected': 'a'}
            ]
        ),
        Problem(
            id="sum_list",
            description="Write 'sum_list(numbers)' that returns sum.",
            difficulty_level=1,
            test_cases=[
                {'input': [1, 2, 3], 'expected': 6},
                {'input': [10, 20], 'expected': 30}
            ],
            hidden_test_cases=[
                {'input': [], 'expected': 0},
                {'input': [-5, 5], 'expected': 0},
                {'input': [100], 'expected': 100}
            ]
        ),
        Problem(
            id="find_max",
            description="Write 'find_max(numbers)' that returns maximum.",
            difficulty_level=2,
            test_cases=[
                {'input': [1, 5, 3], 'expected': 5},
                {'input': [-5, -1], 'expected': -1}
            ],
            hidden_test_cases=[
                {'input': [0], 'expected': 0},
                {'input': [-10, -20, -5], 'expected': -5},
                {'input': [100, 200, 50], 'expected': 200}
            ]
        ),
        Problem(
            id="count_vowels",
            description="Write 'count_vowels(text)' that counts vowels.",
            difficulty_level=2,
            test_cases=[
                {'input': 'hello', 'expected': 2},
                {'input': 'aeiou', 'expected': 5}
            ],
            hidden_test_cases=[
                {'input': 'xyz', 'expected': 0},
                {'input': 'AEIOU', 'expected': 5},
                {'input': '', 'expected': 0}
            ]
        ),
        Problem(
            id="is_palindrome",
            description="Write 'is_palindrome(s)' that checks palindrome.",
            difficulty_level=3,
            test_cases=[
                {'input': 'racecar', 'expected': True},
                {'input': 'hello', 'expected': False}
            ],
            hidden_test_cases=[
                {'input': 'a', 'expected': True},
                {'input': '', 'expected': True},
                {'input': 'abba', 'expected': True}
            ]
        ),
        Problem(
            id="fibonacci",
            description="Write 'fibonacci(n)' efficiently.",
            difficulty_level=4,
            test_cases=[
                {'input': 0, 'expected': 0},
                {'input': 5, 'expected': 5}
            ],
            hidden_test_cases=[
                {'input': 1, 'expected': 1},
                {'input': 10, 'expected': 55},
                {'input': 7, 'expected': 13}
            ]
        ),
        Problem(
            id="longest_substring",
            description="Write 'longest_substring(s)' for longest unique substring length.",
            difficulty_level=5,
            test_cases=[
                {'input': 'abcabcbb', 'expected': 3},
                {'input': 'bbbbb', 'expected': 1}
            ],
            hidden_test_cases=[
                {'input': 'pwwkew', 'expected': 3},
                {'input': '', 'expected': 0},
                {'input': 'abcdef', 'expected': 6}
            ]
        )
    ]

    print(f"✓ {len(problem_pool)} problems loaded (with hidden tests)")

    # 단일 실험 실행
    print("\n[Option 1] Single experiment")
    print("[Option 2] Multiple experiments (3 runs for statistical significance)")
    choice = input("\nSelect option (1 or 2): ").strip()

    if choice == "2":
        run_multiple_experiments(problem_pool, num_runs=3, iterations=20)
    else:
        generator = create_openai_generator()
        print("✓ Generator ready")

        framework = ModularOctadFramework(problem_pool, generator)
        print("✓ Framework initialized\n")

        framework.run(iterations=20)
        framework.save_results("octad_exp1")

    print("\n✓ Complete!")


if __name__ == "__main__":
    main()