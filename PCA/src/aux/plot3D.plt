set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb "grey" behind
unset border
unset xtics
unset ytics
unset ztics
set view equal xyz
set xyplane 0
set style line 50 lt 1 lc rgb "red"   lw 3
set style line 51 lt 1 lc rgb "green" lw 3
set style line 52 lt 1 lc rgb "blue"  lw 3
set arrow 1 from 0,0,0 to 256,0,0 empty ls 50
set arrow 2 from 0,0,0 to 0,256,0 empty ls 51
set arrow 3 from 0,0,0 to 0,0,256 empty ls 52
set view 50, 20
splot 'aux/datapoints.dat' using 1:2:3:4 with points pt 7 ps 2 lc rgb variable notitle
