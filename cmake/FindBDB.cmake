# BDB_INCLUDE_DIR
# BDB_LIBRARIES

FIND_PATH(BDB_INCLUDE_DIR NAMES db_cxx.h db.h
	PATHS 
	/usr/include
	${BDB_ROOT}/include
)
FIND_LIBRARY(BDB_LIBRARY NAMES libdb.a libdb.so libdb.lib db.lib
	PATHS
	/usr/lib/
	/usr/lib/x86_64-linux-gnu/
	${BDB_ROOT}/lib
)

FIND_LIBRARY(BDB_LIBRARY_CXX NAMES libdb_cxx.so libdb_cxx.lib db_cxx.lib
	PATHS
	/usr/lib/
	/usr/lib/x86_64-linux-gnu/
	${BDB_ROOT}/lib
)

IF(BDB_INCLUDE_DIR)
IF(BDB_LIBRARY)
IF(BDB_LIBRARY_CXX)
	SET(BDB_LIBRARIES ${BDB_LIBRARY} ${BDB_LIBRARY_CXX})
	SET(BDB_FOUND "YES")
ENDIF(BDB_LIBRARY_CXX)
ENDIF(BDB_LIBRARY)
ENDIF(BDB_INCLUDE_DIR)

MARK_AS_ADVANCED(
	BDB_LIBRARY BDB_LIBRARY_CXX BDB_LIBRARIES
)
	
	

