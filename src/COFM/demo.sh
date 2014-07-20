#/bin/sh

make
if [ $? -ne 0 ]; then
    exit 1
fi
./run -d 0 -r True
#./run -d 0 -r False

#valgrind --tool=memcheck --leak-check=full --show-reachable=yes --error-limit=no --log-file=valgrind.log ./demo.sh
