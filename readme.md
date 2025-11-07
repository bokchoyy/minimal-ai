# CS-171 Checkers Project
Current Version: 1.1.0.11052019
# 10/28/2019
Fixed a bug in **python** shell that Board parameter should be (col,row) instead of (row,col) 
# 10/29/2019
Fixed java copy constructor
# 10/30/2019
Fixed **cpp/java** undo function causing the wrong attributes of checkers. <br>
Fixed **cpp/java** undo black counter and white counter.<br>
Fixed **java** undo function (equal problem)<br>
Make counters in **cpp/java/python** more stable.<br>
# 11/05/2019
Added a new feature (**cpp/java/python**): "self play mode" which can be used to debug more efficiently.
# 11/14/2019
This update is about Poor AI:
1. Added support for 3.6.8
2. Fix some problem that causes Poor AI crashing in some rare cases.

# 11/30/2019
Added the Good AI.

# 12/05/2019
Blocked some invalid moves.
-----------------------------------------------------------------------------------------------------------------------------
# Minimal AI — Checkers Project
Goal: Beat or tie Random AI ≥ 60% on a 7×7, k=2 board  
Result: Achieved 90% Win+Tie rate

# How to Run
From your OpenLab or local terminal:

```bash
cd src/checkers-python
python3 main.py 7 7 2 l ../../Tools/Sample_AIs/Minimal_AI/main.py ../../Tools/Sample_AIs/Random_AI/main.py

# To measure win+tie rate:
//Use this code in ur terminal!
wins=0; ties=0; total=40
for i in $(seq 1 $total); do
  out=$(python3 main.py 7 7 2 l ../../Tools/Sample_AIs/Minimal_AI/main.py ../../Tools/Sample_AIs/Random_AI/main.py 2>/dev/null | grep -E 'player [12] wins|Tie' | tail -n1)
  if echo "$out" | grep -q 'player 1 wins'; then wins=$((wins+1)); fi
  if echo "$out" | grep -q 'Tie'; then ties=$((ties+1)); fi
done
echo "Win+Tie rate: $(( (wins+ties)*100/total ))%"
