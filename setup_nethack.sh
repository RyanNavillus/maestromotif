FILE_PATH=$(python -c "import nle.nethack.nethack; print(nle.nethack.nethack.__file__)")

# Define fix to self.options variable
NEW_LINE="        self.options = list(options) + [\"name:\" + playername] if options[0][0] != \"@\" else list(options)"

# Use sed to replace line 202
sed -i "202s/.*/$NEW_LINE/" "$FILE_PATH"

# Line number to remove
line_number=210
# Use sed to remove the specified line
sed -i "${line_number}d" $FILE_PATH
sed -i "${line_number}d" $FILE_PATH

# File to modify
FILE_PATH=$(python -c "import nle.env.base; print(nle.env.base.__file__)")

# Line number to insert code
line_number=285

# Code to fix additional variables references
code_to_insert='        self._inv_strs_index = self._observation_keys.index("inv_strs")
        self._inv_letters_index = self._observation_keys.index("inv_letters")
        self._inv_oclasses_index = self._observation_keys.index("inv_oclasses")'

# Use awk to insert the code
awk -v n="$line_number" -v code="$code_to_insert" 'NR == n {print code} {print}' $FILE_PATH > tmp_file && mv tmp_file $FILE_PATH
