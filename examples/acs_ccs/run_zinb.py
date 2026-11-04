"""Convenience runner: FACTMx zinb structured RNA head."""
import sys
from acs_ccs_runner import parse_args, main

if __name__ == '__main__':
    sys.argv.extend(['--model', 'zinb'])
    main(parse_args())
