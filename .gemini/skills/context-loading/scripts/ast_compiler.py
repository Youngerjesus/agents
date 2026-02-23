import ast
import sys
import os
from typing import List, Tuple

def read_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {e}"

def extract_skeleton(code: str) -> str:
    """
    Parses the code and returns a skeleton with signatures and docstrings only.
    Implementation details are replaced with '...'.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code # Fallback for non-Python files or syntax errors

    class SkeletonVisitor(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Keep docstring
            docstring = ast.get_docstring(node)

            # Create a new body with just the docstring (if exists) and Ellipsis (...)
            new_body = []
            if docstring:
                new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
            new_body.append(ast.Expr(value=ast.Constant(value=...)))

            node.body = new_body
            return node

        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            # Process methods within the class
            self.generic_visit(node)
            return node

    visitor = SkeletonVisitor()
    new_tree = visitor.visit(tree)
    return ast.unparse(new_tree)

def generate_xml(targets: List[str], refs: List[str], note: str = "") -> str:
    xml_output = ["<context_document>"]

    for path in targets:
        content = read_file(path)
        xml_output.append(f'    <file path="{path}" type="target_editable">')
        xml_output.append('        <![CDATA[')
        xml_output.append(content)
        xml_output.append('        ]]>')
        xml_output.append('    </file>')

    for path in refs:
        content = read_file(path)
        if path.endswith('.py'):
            content = extract_skeleton(content)

        xml_output.append(f'    <file path="{path}" type="reference_readonly">')
        xml_output.append('        <![CDATA[')
        xml_output.append(content)
        xml_output.append('        ]]>')
        xml_output.append('    </file>')

    if note:
        xml_output.append('    <architect_note>')
        xml_output.append(f'        {note}')
        xml_output.append('    </architect_note>')

    xml_output.append("</context_document>")
    return "\n".join(xml_output)

def main():
    """
    Usage: python ast_compiler.py --targets file1.py file2.py --refs file3.py --note "Instructions" --output contexts/my_context.xml
    """
    import argparse
    parser = argparse.ArgumentParser(description='Generate Context XML')
    parser.add_argument('--targets', nargs='*', default=[], help='List of target files')
    parser.add_argument('--refs', nargs='*', default=[], help='List of reference files')
    parser.add_argument('--note', type=str, default="", help='Architect note')
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    xml_content = generate_xml(args.targets, args.refs, args.note)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        print(f"Context XML saved to {args.output}")
    else:
        print(xml_content)

if __name__ == "__main__":
    main()
