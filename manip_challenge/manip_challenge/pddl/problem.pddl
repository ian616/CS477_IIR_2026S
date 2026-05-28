(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    hammer - item
    table left_storage right_storage bookshelf buffer1 buffer2 - location
  )

  (:init
    (at hammer table)
    (buffer buffer1)
    (buffer buffer2)
    (buffer-free buffer1)
    (buffer-free buffer2)
    (clear hammer)
    (goal-at hammer right_storage)
    (graspable hammer)
    (handempty)
    (safe hammer)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target hammer)
  )

  (:goal
    (and
      (at hammer right_storage)
    )
  )
)
