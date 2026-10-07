# mapgen — 휴리스틱은 고정, 맵을 PPO로 설계한다

술래·도망자는 기존 휴리스틱(`ChasePolicy` / `FleePolicy`)을 그대로 쓰고,
**PPO 설계자**가 빈 공터(또는 Lv3)에서 구조물을 하나씩 놓아 목표 승률 τ와
시간초과 제약(≤ 20%, 한 판 90초)을 맞춘다. `tag_game/` 엔진은 수정하지 않았다.

결과와 해석은 [`REPORT.md`](REPORT.md).

## 구성

| 파일 | 역할 |
|---|---|
| `engine.py` | `TeamWorld`: N 술래 vs M 도망자 + 맵 장치(덤불 `B`, 아이템 패드 `I`, 워프 숫자 쌍). `TeamController`: 휴리스틱에 상대 선택·아이템 우회만 얹음. `play()` 한 판 |
| `sim.py` | 병렬 휴리스틱 대전, 같은 시드 묶음(공통 난수) |
| `design.py` | 설계 MDP: 25×15 캔버스, 8종 구조물 × 299칸 + STOP = 2,393 행동, numpy 행동 마스크, 퍼텐셜 보상 |
| `gen_data.py` | 랜덤 구성 맵 + 실제 대전 → 서로게이트 학습 데이터 |
| `surrogate.py` | CNN 앙상블: 맵 → (승률, 시간초과율, 평균 시간) |
| `train_designer.py` | MaskablePPO + 공간 행동 헤드(칸마다 로짓), 서로게이트 점수를 16개 env 한 배치로 계산 |
| `greedy.py` | 서로게이트 그리디 탐색 — 학습 없는 기준선이자 도달 가능성 점검 |
| `validate.py` | 설계자 · 그리디 · 랜덤 탐색 맵을 **보류 시드 실제 대전**으로 채점 |
| `analyze.py` | 구조 특성 ↔ 승률 상관, 1v1 vs 2v4 비교, 방법 간 검정, 그림 |
| `team_size.py` | 술래 2 vs 도망자 2/4/6: 기준 맵과 설계자 맵의 인원별 승률 추이 |
| `replay.py` | 구성 과정 리플레이 HTML |

## 규칙 (모드)

| 모드 | 술래 | 도망자 | 맵의 발전기 | 해금에 필요 | 도망자 팀 승리 |
|---|---|---|---|---|---|
| `1v1` | 1 | 1 | 5 | 3 | 1명 탈출 |
| `2v4` | 2 | 4 | 7 | 5 | 2명 탈출 (잡힌 도망자는 퇴장, 여럿이 한 발전기를 돌리면 속도 합산) |
| `2v2` · `2v6` | 2 | 2 · 6 | 7 | 5 | 1명 · 3명 탈출 (인원 추이 분석 전용, 설계자 학습에는 쓰지 않음) |

공통: 한 판 90초, 출구 후보 4개 중 3개가 해금 순간에 추첨(`late_exits`), 스폰 랜덤,
시간초과 = 술래 승.

## 실행 순서

```bash
python tests/test_mapgen.py
python -m mapgen.gen_data --maps-1v1 1500 --maps-2v4 1000 --games 16 --minutes 15
python -m mapgen.surrogate --data mapgen/data/random.jsonl --out mapgen/runs/surrogate_r0.pt
python -m mapgen.train_designer --surrogate mapgen/runs/surrogate_r0.pt --steps 60000 --out mapgen/runs/designer_r0
python -m mapgen.validate --designer mapgen/runs/designer_r0/designer.zip --surrogate mapgen/runs/surrogate_r0.pt --tag r0
# 셀프 피드백 라운드: 검증 맵(실제 라벨)을 데이터에 더해 서로게이트 재학습 → 설계자 미세조정 → 새 시드로 재검증
python -m mapgen.surrogate --data mapgen/data/random.jsonl mapgen/results/validate_r0.jsonl --out mapgen/runs/surrogate_r1.pt
python -m mapgen.train_designer --surrogate mapgen/runs/surrogate_r1.pt --init mapgen/runs/designer_r0/designer.zip --steps 40000 --out mapgen/runs/designer_r1
python -m mapgen.validate --designer mapgen/runs/designer_r1/designer.zip --surrogate mapgen/runs/surrogate_r1.pt --tag r1 --seed0 200000
python -m mapgen.analyze --data mapgen/data/random.jsonl --validate mapgen/results/validate_r0.jsonl mapgen/results/validate_r1.jsonl
python -m mapgen.replay --validate mapgen/results/validate_r1.jsonl --out mapgen/results/replay.html
```

Windows 콘솔에서는 `PYTHONIOENCODING=utf-8`을 붙인다.
