




################################
import os
import json

def load_schema(schema_file='schema.json'):
    """Load schema from JSON file"""
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file '{schema_file}' not found. Create schema.json first!")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in schema file")

def get_table_schema(schema, table_name):
    for table in schema:
        if table["table_name"] == table_name:
            return table
    raise ValueError(f"Table {table_name} not found")

def encode_record(record_dict, table_name, schema):
    table_schema = get_table_schema(schema, table_name)
    fields = table_schema["fields"]
    
    record_bytes = b""
    
    for field in fields:
        name = field["name"]
        ftype = field["type"]
        value = record_dict.get(name)
        
        if ftype == "int":
            record_bytes += value.to_bytes(4, byteorder='little', signed=True)
            
        elif ftype == "float":
            # FIXED: Convert float to int representation
            float_as_int = int(value)
            record_bytes += float_as_int.to_bytes(4, byteorder='little', signed=True)
            
        elif ftype.startswith("char("):
            length = int(ftype[5:-1])
            encoded_str = str(value).encode("utf-8")
            padded_str = encoded_str[:length].ljust(length, b'\x00')
            record_bytes += padded_str
            
        elif ftype.startswith("varchar("):
            max_length = int(ftype[8:-1])
            encoded_str = str(value).encode("utf-8")
            actual_length = min(len(encoded_str), max_length)
            record_bytes += bytes([actual_length])
            record_bytes += encoded_str[:actual_length]
            
    return record_bytes

def decode_record(record_bytes, table_name, schema):
    table_schema = get_table_schema(schema, table_name)
    fields = table_schema["fields"]
    
    result_dict = {}
    byte_offset = 0
    
    for field in fields:
        name = field["name"]
        ftype = field["type"]
        
        if ftype == "int":
            int_bytes = record_bytes[byte_offset:byte_offset + 4]
            value = int.from_bytes(int_bytes, byteorder='little', signed=True)
            byte_offset += 4
            
        elif ftype == "float":
            float_bytes = record_bytes[byte_offset:byte_offset + 4]
            value = int.from_bytes(float_bytes, byteorder='little', signed=True)
            byte_offset += 4
            
        elif ftype.startswith("char("):
            length = int(ftype[5:-1])
            char_bytes = record_bytes[byte_offset:byte_offset + length]
            value = char_bytes.rstrip(b'\x00').decode('utf-8')
            byte_offset += length
            
        elif ftype.startswith("varchar("):
            length_byte = record_bytes[byte_offset]
            byte_offset += 1
            max_length = int(ftype[8:-1])
            actual_length = min(length_byte, max_length)
            var_bytes = record_bytes[byte_offset:byte_offset + actual_length]
            value = var_bytes.decode('utf-8')
            byte_offset += actual_length
            
        result_dict[name] = value
        
    return result_dict


def insert_structured_record(table_name, schema, record_dict):
    table_schema = get_table_schema(schema, table_name)
    file_path = table_schema["file_name"]
    
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    
    encoded_record = encode_record(record_dict, table_name, schema)
    
    with open(file_path, 'ab') as heap_file:
        heap_file.write(encoded_record)
    
    print(f"Inserted record into {table_name} heap file: {file_path}")

def read_all_structured_records(table_name, schema):
    table_schema = get_table_schema(schema, table_name)
    file_path = table_schema["file_name"]
    
    if not os.path.exists(file_path):
        return []
    
    record_size = 0
    for field in table_schema["fields"]:
        ftype = field["type"]
        if ftype == "int" or ftype == "float":
            record_size += 4
        elif ftype.startswith("char("):
            record_size += int(ftype[5:-1])
        elif ftype.startswith("varchar("):
            record_size += 1 + int(ftype[8:-1])
    
    records = []
    with open(file_path, "rb") as f:
        while True:
            record_bytes = f.read(record_size)
            if len(record_bytes) < record_size:
                break
            record = decode_record(record_bytes, table_name, schema)
            records.append(record)
    
    return records


################################


def parse_select_query(query, schema) -> dict:
    """
    Parse a simple SELECT query of form:
    SELECT field_list FROM table_name WHERE field = value
    Example output:
    {
        "fields": ["name", "salary"],  # or ["*"] for all
        "table": "Employee",
        "condition": {"field": "id", "value": 3}  # or None if no WHERE
    }
    """
    query = query.strip().rstrip(';')
    # Basic split
    parts = query.split()
    result = {"fields": [], "table": None, "condition": None}

    # Get fields (after SELECT, before FROM)
    select_index = parts.index("SELECT")
    from_index = parts.index("FROM")
    fields_part = ' '.join(parts[select_index + 1:from_index]).strip()
    if fields_part == '*':
        result["fields"] = ["*"]
    else:
        result["fields"] = [f.strip() for f in fields_part.split(',')]

    # Get table name (after FROM)
    table_name = parts[from_index + 1]
    result["table"] = table_name

    # Check for WHERE clause
    if "WHERE" in parts:
        where_index = parts.index("WHERE")
        # Assume condition like field = value (no complex parsing)
        cond_field = parts[where_index + 1]
        cond_op = parts[where_index + 2]
        cond_value = parts[where_index + 3]

        # Remove quotes from value if exists
        if cond_value.startswith("'") and cond_value.endswith("'"):
            cond_value = cond_value[1:-1]

        # Convert to int or float if applicable based on schema
        for table in schema:
            if table["table_name"] == table_name:
                for field in table["fields"]:
                    if field["name"] == cond_field:
                        field_type = field["type"]
                        if field_type.startswith("int"):
                            cond_value = int(cond_value)
                        elif field_type.startswith("float"):
                            cond_value = float(cond_value)
                        break
                break

        result["condition"] = {"field": cond_field, "op": cond_op, "value": cond_value}

    return result


def parse_insert_query(query, schema) -> dict:
    """
    Parse a simple INSERT query of form:
    INSERT INTO table_name (field1, field2, ...) VALUES (value1, value2, ...)
    Example output:
    {
        "table": "Employee",
        "fields": ["id", "name", "salary"],
        "values": [4, "Alice", 4500]
    }
    """
    query = query.strip().rstrip(';')
    # Basic parsing approach:
    # Split into parts: before VALUES and after
    before_values, after_values = query.split("VALUES")
    before_values = before_values.strip()
    after_values = after_values.strip()

    # Get table name
    parts = before_values.split()
    table_name = parts[2]  # after INSERT INTO

    fields_str = before_values[before_values.find("(")+1 : before_values.find(")")]
    fields = [f.strip() for f in fields_str.split(",")]

    values_str = after_values[after_values.find("(")+1 : after_values.find(")")]
    raw_values = [v.strip() for v in values_str.split(",")]

    # Convert values based on schema types
    values = []
    for i, val in enumerate(raw_values):
        val = val.strip()
        # Remove quotes for strings
        if val.startswith("'") and val.endswith("'"):
            val = val[1:-1]
        else:
            # Convert to int or float if applicable
            # Find type from schema
            for table in schema:
                if table["table_name"] == table_name:
                    field_type = table["fields"][i]["type"]
                    if field_type.startswith("int"):
                        val = int(val)
                    elif field_type.startswith("float"):
                        val = float(val)
                    break
        values.append(val)

    return {"table": table_name, "fields": fields, "values": values}


def execute_query(query, schema):
    """
    Execute a SELECT or INSERT query on the structured records stored in the heap file.
    For this example, reading/writing heap file is assumed through simple functions:
    - read_all_structured_records(table_name, schema): returns list of dict records
    - insert_structured_record(table_name, schema, record_dict): inserts record
    """

    q = query.strip().lower()
    if q.startswith("select"):
        parsed = parse_select_query(query, schema)
        table = parsed["table"]
        fields = parsed["fields"]
        condition = parsed["condition"]

        # Read all records
        records = read_all_structured_records(table, schema)

        # Filter records by condition
        if condition:
            cond_field = condition["field"]
            cond_value = condition["value"]
            cond_op = condition.get("op", "=")
            filtered = []
            for rec in records:
                val = rec.get(cond_field)
                if cond_op == "=" and val == cond_value:
                    filtered.append(rec)
                elif cond_op == ">" and val > cond_value:
                    filtered.append(rec)
                elif cond_op == "<" and val < cond_value:
                    filtered.append(rec)
                # Add more ops as needed
            records = filtered

        # Select requested fields
        if fields == ["*"]:
            return records
        else:
            result = []
            for rec in records:
                selected = {f: rec.get(f) for f in fields}
                result.append(selected)
            return result

    elif q.startswith("insert"):
        parsed = parse_insert_query(query, schema)
        table = parsed["table"]
        fields = parsed["fields"]
        values = parsed["values"]
        record_dict = dict(zip(fields, values))
        insert_structured_record(table, schema, record_dict)
        return {"message": "Record inserted."}

    else:
        return {"error": "Unsupported query type"}


# Test schema (add this at top after imports)
schema = [
    {
        "table_name": "Employee",
        "file_name": "employee.dat",
        "fields": [
            {"name": "id", "type": "int"},
            {"name": "name", "type": "char(20)"},
            {"name": "salary", "type": "float"}
        ]
    }
]

# SIMPLE TEST FUNCTIONS (add these - they simulate heap file for testing)
def read_all_structured_records(table_name, schema):
    print(f"[DEBUG] Reading from {table_name} heap file (simulated)")
    return []  # Empty for now - will populate after INSERT

def insert_structured_record(table_name, schema, record_dict):
    print(f"[DEBUG] Inserting into {table_name}: {record_dict}")

# TEST QUERIES (add at BOTTOM of file)
if __name__ == "__main__":
    print("=== Testing Lab 03 Query Processor ===\n")
    
    # Test 1: Parse SELECT
    select_query = "SELECT name, salary FROM Employee WHERE id = 3;"
    parsed_select = parse_select_query(select_query, schema)
    print("1. Parsed SELECT:", parsed_select)
    
    # Test 2: Parse INSERT  
    insert_query = "INSERT INTO Employee (id, name, salary) VALUES (4, 'Alice', 4500);"
    parsed_insert = parse_insert_query(insert_query, schema)
    print("\n2. Parsed INSERT:", parsed_insert)
    
    # Test 3: Execute queries
    print("\n3. Executing SELECT:", execute_query(select_query, schema))
    print("4. Executing INSERT:", execute_query(insert_query, schema))


    # Test 1: INSERT first (create data)
    insert_result = execute_query("INSERT INTO Employee (id, name, salary) VALUES (1, 'Bob', 3000);", schema)
    print("INSERT 1:", insert_result)
    
    insert_result = execute_query("INSERT INTO Employee (id, name, salary) VALUES (2, 'Alice', 4500);", schema)
    print("INSERT 2:", insert_result)
    
    # Test 2: SELECT with condition
    select_result = execute_query("SELECT * FROM Employee WHERE id = 2;", schema)
    print("\nSELECT id=2:", select_result)
    
    # Test 3: SELECT specific fields
    select_result = execute_query("SELECT name, salary FROM Employee WHERE salary > 3500;", schema)
    print("SELECT salary>3500:", select_result)