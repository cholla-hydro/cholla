#!/bin/bash

FQDN=$(hostname --fqdn)

#-- Summit
if [[ $FQDN == *"summit."* ]]; then
  echo "summit"
fi
