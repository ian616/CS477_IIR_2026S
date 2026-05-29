(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    strawberry - item
    table left_storage right_storage bookshelf buffer1 buffer2 - location
  )

  (:init
    (at strawberry table)
    (buffer buffer1)
    (buffer buffer2)
    (buffer-free buffer1)
    (buffer-free buffer2)
    (goal-at strawberry right_storage)
    (handempty)
    (safe strawberry)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target strawberry)
  )

  (:goal
    (and
      (at strawberry right_storage)
    )
  )
)
