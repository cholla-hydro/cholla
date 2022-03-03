#!/bin/bash

if [ ${CHOLLA_MACHINE} ]; then
  echo ${CHOLLA_MACHINE}
  exit 0
fi

FQDN=$(hostname --fqdn)

<<<<<<< HEAD
if [[ $FQDN == *"c3po."* ]] ; then
  echo "c3po"
  exit 0
fi


echo "unknown"
exit 1
=======
case $FQDN in
  *summit* | *peak*)
    echo "summit"
    exit 0 ;;
  poplar.* | tulip.*)
    echo "poplar"
    exit 0 ;;
  *crc.*)
    echo "crc"
    exit 0 ;;
  *spock* | birch*)
    echo "spock"
    exit 0 ;;
  *c3po* )
    echo "c3po"
    exit 0 ;;
  *crusher* | *frontier* )
    echo "frontier"
    exit 0 ;;
  *)
    host=$(hostname)
    echo "Using default hostname, expecting make.host.$host" >&2
    sleep 1
    echo `hostname`
    exit 0
esac
>>>>>>> 94fdc498101d98279008e1e6a7ba0ae030da23cd
