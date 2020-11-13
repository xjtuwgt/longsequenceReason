#!/bin/sh
JOBS_PATH=hotpot_jobs
LOGS_PATH=logs
for ENTRY in "${JOBS_PATH}"/*.sh; do
  chmod +x $ENTRY
  FILE_NAME="$(basename "$ENTRY")"
  echo $FILE_NAME
#  /mnt/cephfs2/asr/users/ming.tu/software/kaldi/egs/wsj/s5/utils/queue.pl -q g2.q -l gpu=1 $LOGS_PATH/$FILE_NAME.log $ENTRY &
#  sleep 3
  #/mnt/cephfs2/asr/users/ming.tu/software/kaldi/egs/wsj/s5/utils/queue.pl -q g.q -l gpu=4 -l h=GPU_10_252_192_[8-9]*  $LOGS_PATH/$FILE_NAME.log $ENTRY &
  /mnt/cephfs2/asr/users/ming.tu/software/kaldi/egs/wsj/s5/utils/queue.pl -q g.q -l gpu=4 $LOGS_PATH/$FILE_NAME.log $ENTRY &
  sleep 20
done