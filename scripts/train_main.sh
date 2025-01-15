k_shots=(5 10 20 50)

for k_shot in "${k_shots[@]}"
do

  python main.py  --task scierc  --k_sample $k_shot --use_knn True
done

k_shots=(5 10 20 50)
for k_shot in "${k_shots[@]}"
do

  python main.py  --task scierc  --k_sample $k_shot

done