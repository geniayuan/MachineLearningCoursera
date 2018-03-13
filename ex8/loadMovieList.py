#GETMOVIELIST reads the fixed movie list in movie.txt and returns a
#cell array of the words
#   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt
#   and returns a cell array of the words in movieList.

def loadMovieList():
    movieList = []
    ## Read the fixed movieulary list
    with open('movie_ids.txt') as fid:
        for line in fid:
            finfo = line.strip().split(' ')
            fname = " ".join(finfo[1:])
            movieList.append(fname)

    return movieList
