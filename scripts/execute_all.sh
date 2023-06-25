SCRIPTS_DIR="./scripts"

# Change to the scripts directory
cd "$SCRIPTS_DIR"

# Execute each script
for script in *.sh; do
    if [ -x "$script" ]; then
        echo "Executing script: $script"
        bash "$script"
    else
        echo "No execution permission for script: $script"
    fi
done