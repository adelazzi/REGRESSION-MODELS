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

# MAIN - Load schema from file
# Load schema from JSON file
schema = load_schema('LABO2/schema.json')

record = {"id": 12, "name": "Alice", "salary": 50000.0}
print("Original:", record)

encoded = encode_record(record, "Employee", schema)
decoded = decode_record(encoded, "Employee", schema)
print("Decoded:", decoded)

insert_structured_record("Employee", schema, record)
all_records = read_all_structured_records("Employee", schema)
print("All records:", all_records)
