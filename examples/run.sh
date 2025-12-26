#!/bin/bash


if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1;
fi

echo "$0: download PartialSpoof evaluation label"
mkdir -p PartialSpoof
if [ ! -f PartialSpoof/label_PartialSpoof_eval.txt ]; then wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.1.0/label_PartialSpoof_eval.txt -P PartialSpoof; fi

echo "$0: download example scores"
mkdir -p scores
if [ ! -f scores/baseline_e55_ps-eval.tgz ]; then wget https://github.com/hieuthi/partialspoof-metrics/releases/download/v1.0.0/baseline_e55_ps-eval.tgz -P scores; fi
if [ ! -d scores/baseline_e55_ps-eval ]; then tar -xvf scores/baseline_e55_ps-eval.tgz -C scores/ ; fi

labelfile=PartialSpoof/label_PartialSpoof_eval.txt
resultdir=baseline_e55_ps-eval

mkdir -p results

echo "$0: calculate utterance-based EER"
python ../calculate_eer.py --labpath ${labelfile} \
                          --scopath scores/${resultdir}/utt.score \
                          --savepath results/${resultdir}_utt \
                          --scoreindex 2

echo "$0: Draw score distribution dentisy figure and save at results/${resultdir}_utt/score.pdf"
python ../draw_score_distribution.py --loadpath results/${resultdir}_utt \
                                    --savepath results/${resultdir}_utt/score.pdf \
                                    --threshold 0.5 \
                                    --xmin -0.5 \
                                    --xmax 1.5

for unit in 0.02 0.04 0.08 0.16 0.32 0.64; do
  echo "$0: calculate $unit frame-based EER"
  python ../calculate_eer.py --labpath ${labelfile} \
                          --scopath scores/${resultdir}/unit${unit}.score \
                          --savepath results/${resultdir}_${unit} \
                          --unit ${unit} \
                          --scoreindex 3 

  echo "$0: calculate Utterance-based EER upscaled from ${unit}s score"
  python ../calculate_eer.py --labpath ${labelfile} \
                          --scopath scores/${resultdir}/unit${unit}.score \
                          --savepath results/${resultdir}_utt${unit} \
                          --unit ${unit} \
                          --scoreindex 3 \
                          --zoom 0

  echo "$0: calculate millisecond EER from ${unit}s score"
  python ../calculate_mseer.py --labpath ${labelfile} \
                          --scopath scores/${resultdir}/unit${unit}.score \
                          --savepath results/${resultdir}_ms${unit} \
                          --unit ${unit} \
                          --scoreindex 3
done


echo "==== Result Summary ===="
echo
eer=$( grep "eer=" results/${resultdir}_utt/result.txt | awk -F"=" '{ printf "%.2f", $2*100}')
echo "Utterance EER: ${eer}%"
printf "Utterance EER Threshold: "
python ../calculate_accuracy.py --loadpath results/${resultdir}_utt/ --eer_threshold
printf "95%% Recall Threshold: "
python ../calculate_accuracy.py --loadpath results/${resultdir}_utt/ --recall 0.95
printf "95%% Precision Threshold: "
python ../calculate_accuracy.py --loadpath results/${resultdir}_utt/ --precision 0.95


echo
echo "Frame-based EER"
for unit in 0.02 0.04 0.08 0.16 0.32 0.64; do
  printf "${unit}s\t"
done
echo
for unit in 0.02 0.04 0.08 0.16 0.32 0.64; do
  eer=$( grep "eer=" results/${resultdir}_${unit}/result.txt | awk -F"=" '{ printf "%.2f", $2*100}')
  printf "${eer}\t"
done
echo

echo
echo "Upscaled Utterance-based EER"
for unit in 0.02 0.04 0.08 0.16 0.32 0.64; do
  printf "${unit}s\t"
done
echo
for unit in 0.02 0.04 0.08 0.16 0.32 0.64; do
  eer=$( grep "eer=" results/${resultdir}_utt${unit}/result.txt | awk -F"=" '{ printf "%.2f", $2*100}')
  printf "${eer}\t"
done
echo

echo
echo "Millisecond EER"
for unit in 0.02 0.04 0.08 0.16 0.32 0.64; do
  printf "${unit}s\t"
done
echo
for unit in 0.02 0.04 0.08 0.16 0.32 0.64; do
  eer=$( grep "eer=" results/${resultdir}_ms${unit}/result.txt | awk -F"=" '{ printf "%.2f", $2*100}')
  printf "${eer}\t"
done
echo



