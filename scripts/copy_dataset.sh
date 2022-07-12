src_folder="/u/scr/nlp/wilds/data/fmow_v1.1/"
dst_folder="/scr/biggest/${USER}/"

if [ ! -d "${dst_folder}" ]; then
    echo "Making $dst_folder"
    mkdir -p $dst_folder
fi

if [ ! -d "$dst_folder/fmow_v1.1" ]; then
    echo "Copying $src_folder to $dst_folder..."
    cp -r ${src_folder} ${dst_folder}
fi