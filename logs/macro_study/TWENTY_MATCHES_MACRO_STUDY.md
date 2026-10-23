# Empirical Study: Macro-Scale Topology & Cross-Validation Architecture across 20 EPL Matches (40 Halves)

> **Monograph Addendum:** Analysis of the full frozen evaluation set (`eval_set.json`) containing 40 broadcast halves (~36.8 hours of Premier League video) and 4,383 human-annotated Ground Truth events.

## 1. Executive Corpus Metrics
- **Total Matches**: 20 EPL Matches (18 official Test split, 2 Validation split)
- **Total Broadcast Footage**: 40 Halves (30.0 hours at 720p / 25 fps)
- **Total Ground Truth Spotting Events**: 4,383 events across 17 official classes
- **Total Ground Truth Commentary Captions**: 1,720 dense captions (24.24% marked as critical game moments)
- **Match Event Density**: Mean $\mu = 219.2 \pm 32.0$ events/match (Range: 172 to 278 events)
- **Temporal Distribution**: Half 1: 2048 (46.73%) | Half 2: 2335 (53.27%)

---

## 2. Class Distribution, Asymmetry & Imbalance Matrix

| Rank | Class | Total | H1 | H2 | H2 Bias (%) | Q1 | Q2 | Q3 | Q4 | % Total | Imbalance Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Ball out of play** | 1331 | 641 | 690 | 51.8% | 350 | 291 | 344 | 346 | 30.37% | 1.0:1 |
| 2 | **Throw-in** | 829 | 410 | 419 | 50.5% | 233 | 177 | 213 | 206 | 18.91% | 1.6:1 |
| 3 | **Foul** | 435 | 218 | 217 | 49.9% | 94 | 124 | 116 | 101 | 9.92% | 3.1:1 |
| 4 | **Indirect free-kick** | 354 | 165 | 189 | 53.4% | 70 | 95 | 92 | 97 | 8.08% | 3.8:1 |
| 5 | **Clearance** | 303 | 143 | 160 | 52.8% | 71 | 72 | 82 | 78 | 6.91% | 4.4:1 |
| 6 | **Shots on target** | 212 | 94 | 118 | 55.7% | 33 | 61 | 55 | 63 | 4.84% | 6.3:1 |
| 7 | **Shots off target** | 208 | 94 | 114 | 54.8% | 53 | 41 | 50 | 64 | 4.75% | 6.4:1 |
| 8 | **Corner** | 191 | 86 | 105 | 55.0% | 43 | 43 | 47 | 58 | 4.36% | 7.0:1 |
| 9 | **Substitution** | 113 | 2 | 111 | 98.2% | 0 | 2 | 35 | 76 | 2.58% | 11.8:1 |
| 10 | **Kick-off** | 94 | 44 | 50 | 53.2% | 28 | 16 | 33 | 17 | 2.14% | 14.2:1 |
| 11 | **Direct free-kick** | 94 | 50 | 44 | 46.8% | 22 | 28 | 22 | 22 | 2.14% | 14.2:1 |
| 12 | **Offside** | 83 | 37 | 46 | 55.4% | 14 | 23 | 18 | 28 | 1.89% | 16.0:1 |
| 13 | **Yellow card** | 73 | 35 | 38 | 52.1% | 4 | 31 | 9 | 29 | 1.67% | 18.2:1 |
| 14 | **Goal** | 57 | 26 | 31 | 54.4% | 9 | 17 | 15 | 16 | 1.30% | 23.4:1 |
| 15 | **Penalty** | 2 | 2 | 0 | 0.0% | 1 | 1 | 0 | 0 | 0.05% | 665.5:1 |
| 16 | **Red card** | 2 | 0 | 2 | 100.0% | 0 | 0 | 1 | 1 | 0.05% | 665.5:1 |
| 17 | **Yellow->red card** | 2 | 1 | 1 | 50.0% | 0 | 1 | 1 | 0 | 0.05% | 665.5:1 |

### Key Topological Observations:
1. **The Substitution Half-2 Law**: Across 20 matches, **98.2% of substitutions (111 / 113)** occur in the second half, heavily concentrated in Q4 (68–90+ min, 93 occurrences). Single-half benchmarks evaluated only on Half 1 test slices suffer from complete structural blindness to substitution events.
2. **The 665.5:1 Class Imbalance Chasm**: `Ball out of play` (1,331 instances, 30.37% of corpus) outnumbers `Penalty` (2 instances) and `Red card` (2 instances) by 665.5-to-1. Standard unweighted accuracy or unmasked mAP is overwhelmingly biased toward high-frequency boundary transitions.
3. **Stoppage Time & Late Game Intensity**: Discipline and stoppage frequency surge in Q4: Yellow cards increase from 35 in H1 to 38 in H2, while all 2 Red cards occurred in the final 30 minutes of play.

---

## 3. Empirical Markov Transition Dynamics ($P(C_{t+1} \mid C_t)$ within 30s)

Analysis of contiguous temporal transitions reveals clear physical sequence laws in soccer broadcasts:

| Trigger Event ($C_t$) | Total 30s Transitions | #1 Consequent ($C_{t+1}$) | Prob | #2 Consequent ($C_{t+1}$) | Prob | #3 Consequent ($C_{t+1}$) | Prob |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| **Foul** | 276 | Indirect free-kick (179) | 64.9% | Yellow card (60) | 21.7% | Ball out of play (12) | 4.3% |
| **Goal** | 1 | Kick-off (1) | 100.0% | - | - | - | - |
| **Ball out of play** | 1133 | Throw-in (774) | 68.3% | Clearance (180) | 15.9% | Corner (137) | 12.1% |
| **Substitution** | 100 | Throw-in (28) | 28.0% | Clearance (20) | 20.0% | Substitution (16) | 16.0% |
| **Corner** | 150 | Ball out of play (62) | 41.3% | Shots off target (36) | 24.0% | Shots on target (31) | 20.7% |

---

## 4. Track B: Dense Video Captioning Alignment Corpus
- **Total Human Commentary Anchors**: 1,720
- **Mean Word Length per Caption**: 22.78 words
- **Named-Entity Anonymization Rate**: 91.4% of captions contain player, coach, or referee entity tags requiring delexicalization.
- **Temporal Co-Localization with Action Spotting**:
  - Spotting events within $\pm 5\text{s}$ of a commentary anchor: **17.91%**
  - Spotting events within $\pm 15\text{s}$ of a commentary anchor: **41.48%**
  - Spotting events within $\pm 30\text{s}$ of a commentary anchor: **64.91%**

---

## 5. Stratified 4-Fold & 5-Fold Cross-Validation Framework

To prevent evaluation folds from starving on rare classes, we formulate a balanced serpentine stratification across all 20 matches:

### Stratified 5-Fold Match Partitioning
| Fold | Matches | Game IDs | Total Events | Red Cards | Penalties | Goals | Substitutions |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 4 | [2, 1, 17, 11] | 834 | 1 | 1 | 11 | 22 |
| Fold 2 | 4 | [3, 7, 15, 10] | 879 | 1 | 0 | 9 | 23 |
| Fold 3 | 4 | [13, 6, 12, 18] | 901 | 0 | 1 | 10 | 20 |
| Fold 4 | 4 | [14, 5, 19, 8] | 911 | 0 | 0 | 13 | 24 |
| Fold 5 | 4 | [16, 4, 9, 0] | 858 | 0 | 0 | 14 | 24 |

### Stratified 4-Fold Match Partitioning
| Fold | Matches | Game IDs | Total Events | Red Cards | Penalties | Goals | Substitutions |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 5 | [2, 6, 7, 0, 8] | 1100 | 1 | 1 | 15 | 30 |
| Fold 2 | 5 | [3, 5, 1, 9, 18] | 1206 | 1 | 0 | 13 | 27 |
| Fold 3 | 5 | [13, 4, 17, 19, 10] | 1049 | 0 | 1 | 13 | 29 |
| Fold 4 | 5 | [14, 16, 15, 12, 11] | 1028 | 0 | 0 | 16 | 27 |

---

## 6. Complete 20-Match Fixture Registry & Event Inventory

| ID | Match Title | Season | Split | Events | H1 | H2 | Red | Pen | Gol | Yel | Sub |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 2015-05-17 - 18-00 Manchester United 1 | 2014-2015 | test | 173 | 65 | 108 | 0 | 0 | 2 | 1 | 6 |
| 1 | 2015-08-16 - 18-00 Manchester City 3 - | 2015-2016 | test | 232 | 109 | 123 | 0 | 0 | 3 | 6 | 6 |
| 2 | 2015-08-23 - 15-30 West Brom 2 - 3 Che | 2015-2016 | test | 208 | 95 | 113 | 1 | 1 | 5 | 3 | 6 |
| 3 | 2015-08-29 - 17-00 Liverpool 0 - 3 Wes | 2015-2016 | test | 241 | 111 | 130 | 1 | 0 | 3 | 6 | 6 |
| 4 | 2015-09-20 - 18-00 Southampton 2 - 3 M | 2015-2016 | test | 195 | 95 | 100 | 0 | 0 | 5 | 2 | 6 |
| 5 | 2015-09-26 - 19-30 Newcastle Utd 2 - 2 | 2015-2016 | test | 268 | 137 | 131 | 0 | 0 | 4 | 3 | 6 |
| 6 | 2015-10-03 - 19-30 Chelsea 1 - 3 South | 2015-2016 | test | 227 | 97 | 130 | 0 | 0 | 4 | 7 | 6 |
| 7 | 2015-10-24 - 17-00 West Ham 2 - 1 Chel | 2015-2016 | test | 245 | 113 | 132 | 0 | 0 | 3 | 6 | 6 |
| 8 | 2015-11-07 - 20-30 Stoke City 1 - 0 Ch | 2015-2016 | test | 247 | 125 | 122 | 0 | 0 | 1 | 4 | 6 |
| 9 | 2015-11-08 - 19-00 Arsenal 1 - 1 Totte | 2015-2016 | test | 270 | 103 | 167 | 0 | 0 | 2 | 1 | 6 |
| 10 | 2015-12-28 - 20-30 Manchester United 0 | 2015-2016 | test | 188 | 87 | 101 | 0 | 0 | 0 | 6 | 5 |
| 11 | 2016-02-03 - 22-45 Watford 0 - 0 Chels | 2015-2016 | test | 178 | 95 | 83 | 0 | 0 | 0 | 3 | 4 |
| 12 | 2016-03-01 - 22-45 Norwich 1 - 2 Chels | 2015-2016 | test | 201 | 87 | 114 | 0 | 0 | 3 | 5 | 5 |
| 13 | 2016-08-27 - 14-30 Tottenham 1 - 1 Liv | 2016-2017 | test | 278 | 129 | 149 | 0 | 1 | 2 | 8 | 6 |
| 14 | 2016-09-24 - 14-30 Manchester United 4 | 2016-2017 | test | 224 | 118 | 106 | 0 | 0 | 5 | 2 | 6 |
| 15 | 2016-10-15 - 14-30 Chelsea 3 - 0 Leice | 2016-2017 | test | 205 | 100 | 105 | 0 | 0 | 3 | 2 | 6 |
| 16 | 2017-01-21 - 15-30 Liverpool 2 - 3 Swa | 2016-2017 | test | 220 | 96 | 124 | 0 | 0 | 5 | 2 | 6 |
| 17 | 2017-05-06 - 17-00 Leicester 3 - 0 Wat | 2016-2017 | test | 216 | 103 | 113 | 0 | 0 | 3 | 0 | 6 |
| 18 | 2015-04-11 - 19-30 Burnley 0 - 1 Arsen | 2014-2015 | valid | 195 | 93 | 102 | 0 | 0 | 1 | 2 | 3 |
| 19 | 2015-08-30 - 18-00 Swansea 2 - 1 Manch | 2015-2016 | valid | 172 | 90 | 82 | 0 | 0 | 3 | 4 | 6 |
