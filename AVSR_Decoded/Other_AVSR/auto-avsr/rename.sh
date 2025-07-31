find ./ -type f -name 'avsr_snr*' -print0 |
while IFS= read -r -d '' file; do
    # Extract the part of the filename after "asr_snr"
    new_name="${file#*/avsr_snr}"
    # Rename the file
    mv "$file" "${file%/*}/$new_name"
done
