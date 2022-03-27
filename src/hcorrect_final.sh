for f in `ls $1/*.txt`; do
mv $f nhsin.txt
./hcorrect_final.o <<!
nhsin.txt
1
out.txt
!
outfile="$1/`basename $f .txt`".out
logfile="$1/`basename $f .txt`".log

script -c "./eigen_correct.o" $logfile <<!
nhsin.txt
`tail -n 1 out.txt | sed 's/ \{1,\}/,/g' | cut -d',' -f2,3,4`
!

mv nhsin.txt $f
mv out.txt $outfile
done
