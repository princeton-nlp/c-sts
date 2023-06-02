#!/bin/bash
if [ $# -ne 2 ]; then
    echo "Error: Please provide the path to the encrypted file and the decryption password."
    echo "Usage: ./extract.sh <path_to_encrypted_file> <password>"
    exit 1
fi
ENCRYPTED_FILE="$1"
PASSWORD="$2"
openssl aes-256-cbc -a -d -salt -pbkdf2 -in "$ENCRYPTED_FILE" -out csts.tar -pass pass:"$PASSWORD"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
	rm -f csts.tar
    echo "Error: Failed to decrypt the file."
    exit $EXIT_CODE
fi
tar -xvf csts.tar
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Failed to extract the decrypted file."
    exit $EXIT_CODE
fi
rm -f csts.tar
if [ $? -ne 0 ]; then
    echo "Error: Failed to remove the files."
    exit $?
fi
echo "Decryption and cleanup completed successfully."
exit 0

