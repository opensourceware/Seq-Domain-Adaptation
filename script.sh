#!/bin/bash

### Set initial time of file
LTIME=`stat -c %Z checkpoint`

while true    
do
   ATIME=`stat -c %Z checkpoint`

   if [[ "$ATIME" != "$LTIME" ]]
   then    
       echo "RUN COMMNAD"
       echo $LTIME
       echo $ATIME
       CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python run_trained_model.py >> output.o
       LTIME=$ATIME
   fi
   sleep 5
done
