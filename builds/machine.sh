#!/bin/bash

FQDN=$(hostname --fqdn)

#-- Summit
if [[ $FQDN == *"summit."* ]]; then
  echo "summit"
  exit 0
fi

if [[ $FQDN == *"poplar."* ]]; then
  echo "poplar"
  exit 0
fi

echo "unknown"
exit 1
