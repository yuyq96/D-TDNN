#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2020   Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0.

. ./cmd.sh
. ./path.sh
set -e
datadir=/home/yuyq/D-TDNN/data/voxceleb1
mfccdir=${datadir}/mfcc
vaddir=${datadir}/mfcc


# The trials file is downloaded by local/make_voxceleb1_v2.pl.
voxceleb1_trials=${datadir}/test/trials
voxceleb1_root=/data/corpora/VoxCeleb1

stage=0

if [ $stage -le 0 ]; then
  # This script creates data/voxceleb1_test for latest version of VoxCeleb1.
  local/make_voxceleb1_v2.pl $voxceleb1_root test ${datadir}/test
fi

if [ $stage -le 1 ]; then
  # Make features and compute the energy-based VAD
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    ${datadir}/test exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh ${datadir}/test
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    ${datadir}/test exp/make_vad $vaddir
  utils/fix_data_dir.sh ${datadir}/test
fi

if [ $stage -le 2 ]; then
  # Apply CMN and removes nonspeech frames
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    ${datadir}/test ${datadir}/test_no_sil ${datadir}/test_no_sil
  utils/fix_data_dir.sh ${datadir}/test_no_sil
fi
