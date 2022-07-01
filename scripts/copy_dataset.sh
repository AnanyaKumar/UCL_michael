machine_name=$(python ./scripts/print_machine.py 2>&1) 
echo "On ${machine_name} for ${USER}"
src_folder="/u/scr/nlp/wilds/data/fmow_v1.1/"
dst_folder="/${machine_name}/scr0/${USER}/"

if [ ! -d "${dst_folder}" ]; then
    mkdir -p $dst_folder
fi

if [ ! -f "$dst_folder/fmow_v1.1" ]; then
    echo "Copying $src_folder to $dst_folder..."
    cp -r ${src_folder} ${dst_folder}
fi