#!/bin/bash

epochs=5
s3_bert="bert-base-uncased"
local_bert="$PRE_TRAINED_HOME/bert/uncased_L-12_H-768_A-12"

for bert_source in s3-bert; do
	if [ "$bert_source" == "s3-bert" ] ; then
		bert_path=$s3_bert
	fi ;
	if [ "$bert_source" == "local-bert" ] ; then
		bert_path=$local_bert
	fi ;
	for nwords in 175; do
		for docrepo in 20news reuters movies; do
			out="./results/$nwords-$bert_source-$docrepo.out"
			echo "PYTHONHASHSEED=0; /usr/bin/time pipenv run python ./tf2_bert_classify.py $bert_path $docrepo $nwords $epochs >& $out"
			PYTHONHASHSEED=0; /usr/bin/time pipenv run python ./tf2_bert_classify.py $bert_path $docrepo $nwords $epochs >& $out
		done
	done
done

