"""Convenience runner: FACTMx nb structured RNA head."""
import sys
from acs_ccs_runner import parse_args, main

if __name__ == '__main__':
    sys.argv.extend(['--model', 'nb'])
    main(parse_args())
