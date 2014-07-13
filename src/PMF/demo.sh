#/bin/sh

make
./run -d 1 -r True -b False

#valgrind --tool=memcheck --leak-check=full --show-reachable=yes --error-limit=no --log-file=valgrind.log ./demo.sh
