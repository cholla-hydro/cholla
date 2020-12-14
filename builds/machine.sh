#!/bin/bash

FQDN=$(hostname --fqdn)

if [ ${CHOLLA_MACHINE} ]; then
  echo ${CHOLLA_MACHINE}
  exit 0
fi

#-- Summit
if [[ $FQDN == *"summit."* ]]; then
  echo "summit"
  exit 0
fi

if [[ $FQDN == *"poplar."* ]] || [[ $FQDN == *"tulip."* ]] ; then
  echo "poplar"
  exit 0
fi

echo "unknown"
exit 1
