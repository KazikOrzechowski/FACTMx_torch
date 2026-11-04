"""Convenience runner: FACTMx topic structured RNA head."""
import sys
from acs_ccs_runner import parse_args, main

if __name__ == '__main__':
    sys.argv.extend(['--model', 'topic'])
    main(parse_args())
