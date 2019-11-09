#!/bin/bash

for nwords in 175 510 full; do 
	for clf in lr svm; do
		for docrepo in movies 20news reuters; do
			for vectorsource in none fasttext; do
				out="./results/$nwords-$clf-$docrepo-$vectorsource.out"
				echo "PYTHONHASHSEED=0 ; /usr/bin/time pipenv run python ./bow_classify.py $clf $docrepo $vectorsource $nwords >& $out"
				PYTHONHASHSEED=0 ; /usr/bin/time pipenv run python ./bow_classify.py $clf $docrepo $vectorsource $nwords >& $out
			done
		done
	done
done

