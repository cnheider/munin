@ECHO OFF

:loop
      ::-------------------------- has argument ?
      if ["%~1"]==[""] (
        ::- echo done.
        set out_path=frozen_env.txt
        goto end
      )
      set out_path=%~1
      ::-------------------------- argument exist ?
      if not exist %~s1 (
        echo not exist
      ) else (
        echo exist
        if exist %~s1\NUL (
          echo is a directory
        ) else (
          echo is a file
        )
      )
      ::--------------------------
      shift
      goto loop
:end

python -m pip freeze --exclude-editable > %out_path%
::- --user


