#!/usr/bin/bash

function usage ()
{
    echo -e "Usage : ./models.sh command \nAvailable commands :\n" \
         "load [name] : loads models saved in [name] dir. Warning : overrides existing models.\n" \
         "save [name] : saves existing models in [name] dir."
    exit 1 
}

if [ $# -eq 0 ]; then
    usage
fi

if [ $1 == "load" ]; then
    cp "$2/generator.h5" "last_generator.h5"
    cp "$2/discriminator.h5" "last_discriminators.h5"
elif [ $1 == "save" ]; then
    mkdir $2
    cp "last_generator.h5" "$2/generator.h5"
    cp "last_discriminator.h5" "$2/discriminator.h5" 
else
    usage
fi

