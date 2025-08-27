# Returns numbers in a sensible format
def nicenumber(number):
	# Very small numbers - use scientific notation
	if abs(number) < 0.1:
		newnumber = str("{:.2E}".format(number))
	# Small numbers - return comma-separated thousands rounded to 2 d.p.
	if abs(number) >= 0.01 and abs(number) < 1E6:
		newnumber = str(f"{number:,.2f}")
	# Large numbers - use scientific notation
	if abs(number) >= 1E6:
		newnumber = str("{:.2E}".format(number))
		
	return newnumber
