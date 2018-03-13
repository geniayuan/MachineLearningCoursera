#PROCESSEMAIL preprocesses a the body of an email and
#returns a list of word_indices
#   word_indices = PROCESSEMAIL(email_contents) preprocesses
#   the body of an email and returns a list of indices of the
#   words contained in the email.
#
import re
import string
import nltk, nltk.stem.porter

def processEmail(email_contents, vocabDict):
    #print("email_contents size:", len(email_contents) )

    # Init return value
    word_indices = []
    # ========================== Preprocess Email ===========================
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    #print("email_contents size:", len(email_contents) )
    # ========================== Tokenize Email ===========================
    # Output the email to screen as well
    print('\n==========Process Email Begin==============')
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', \
                        email_contents)

    print("tokens size:", len(tokens))
    stemmer = nltk.stem.porter.PorterStemmer()

    # Process file
    l = 0

    for token in tokens:

        token = re.sub('[^a-zA-Z0-9]', '', token)
        stemmed = stemmer.stem(token)

        # Skip the word if it is too short
        if stemmed == '' or len(stemmed) < 1:
            continue

        # Print to screen, ensuring that the output lines are not too long
        if (l + len(stemmed) + 1) > 78:
            print(stemmed)
            l = 0
        else:
            print(stemmed, end=" ")
            l = l + len(stemmed) + 1

        if stemmed in vocabDict:
            word_indices.append(vocabDict[stemmed])

    # Print footer
    print('\n\n==========Process Email End==============')

    return word_indices
