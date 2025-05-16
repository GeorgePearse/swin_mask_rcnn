#!/usr/bin/env python3
"""Script to convert print statements to logging in Python files."""
import re
import sys
from pathlib import Path
from typing import List, Tuple

def find_print_statements(file_path: Path) -> List[Tuple[int, str]]:
    """Find all print statements in a file."""
    print_pattern = re.compile(r'^\s*print\s*\(')
    prints = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if print_pattern.match(line):
            prints.append((i + 1, line.strip()))
            
    return prints

def convert_file(file_path: Path, dry_run: bool = True):
    """Convert print statements to logging in a single file."""
    print(f"Processing {file_path}...")
    
    prints = find_print_statements(file_path)
    if not prints:
        print(f"  No print statements found")
        return
        
    print(f"  Found {len(prints)} print statements:")
    for line_num, line in prints:
        print(f"    Line {line_num}: {line}")
    
    if dry_run:
        print("  (Dry run - no changes made)")
    else:
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Add logger import if not present
        has_logger_import = any('from swin_maskrcnn.utils.logging import' in line for line in lines)
        if not has_logger_import:
            # Find the last import line
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    last_import_idx = i
            
            lines.insert(last_import_idx + 1, 'from swin_maskrcnn.utils.logging import get_logger\n')
        
        # Convert print statements
        for line_num, _ in reversed(prints):  # Process in reverse to maintain line numbers
            line_idx = line_num - 1
            line = lines[line_idx]
            
            # Extract the print content
            match = re.match(r'^(\s*)print\s*\((.*)\)\s*$', line)
            if match:
                indent = match.group(1)
                content = match.group(2)
                
                # Determine log level based on content
                if 'error' in content.lower() or 'fail' in content.lower():
                    level = 'error'
                elif 'warn' in content.lower():
                    level = 'warning'
                elif 'debug' in content.lower():
                    level = 'debug'
                else:
                    level = 'info'
                
                # Replace the line
                lines[line_idx] = f'{indent}logger.{level}({content})\n'
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f"  Converted {len(prints)} print statements")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: convert_prints_to_logging.py <file_or_directory> [--dry-run]")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv
    
    if path.is_file():
        convert_file(path, dry_run)
    elif path.is_dir():
        python_files = list(path.rglob('*.py'))
        print(f"Found {len(python_files)} Python files")
        
        files_with_prints = []
        for file_path in python_files:
            if find_print_statements(file_path):
                files_with_prints.append(file_path)
        
        print(f"\nFiles with print statements: {len(files_with_prints)}")
        for file_path in files_with_prints:
            print(f"  {file_path}")
        
        if not dry_run:
            print("\nConverting files...")
            for file_path in files_with_prints:
                convert_file(file_path, dry_run=False)
    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)

if __name__ == '__main__':
    main()