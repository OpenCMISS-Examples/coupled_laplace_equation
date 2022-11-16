PROGRAM ZZZZZZZZ
    
    character*50 :: cline
    
    open (UNIT=15, FILE="resource.txt", STATUS='old',    &
             ACCESS='sequential', FORM='formatted', ACTION='read' )
    read(15, "(A)") cline
    close(15)
    
    open(UNIT=24, FILE="results.txt")
    write(24, "(A)") trim(cline)
    close(24)
    
END PROGRAM ZZZZZZZZ