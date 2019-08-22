#!/usr/bin/perl

my %FUNCTIONS = ( "__slimcuda_gettid" => 1, "__slimcuda_init" => 1 );

use strict;
use warnings;

open(PTXSTUBS, "<ptxstubs.ptx") or die;

my $globals = "static const char[] PTXSTUB_GLOBALS = \"\"\n";
my $infunc = 0;

while(<PTXSTUBS>)
{
    chomp;

    if(/^\.global/) {
        $globals .= "        \"" . $_ . "\\n\"\n";
    }

    if(/^\.visible.* ([A-Za-z0-9_]+)\(/ ) {
        $infunc = (defined $FUNCTIONS{$1}) ? 2: 1;  
        print "$1: $infunc\n";
    }

}
$globals .= ";\n\n";

print "$globals";
