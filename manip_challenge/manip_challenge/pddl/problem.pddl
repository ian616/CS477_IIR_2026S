(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    hammer strawberry - item
    table left_storage right_storage bookshelf buffer1 buffer2 - location
  )

  (:init
    (at hammer table)
    (at strawberry table)
    (buffer buffer1)
    (buffer buffer2)
    (buffer-free buffer1)
    (buffer-free buffer2)
    (clear hammer)
    (goal-at hammer right_storage)
    (goal-at strawberry right_storage)
    (graspable hammer)
    (handempty)
    (safe hammer)
    (safe strawberry)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target hammer)
    (target strawberry)
  )

  (:goal
    (and
      (at strawberry right_storage)
      (at hammer right_storage)
    )
  )
)
