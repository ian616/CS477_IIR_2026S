(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    strawberry_0 strawberry_1 - item
    left_storage right_storage bookshelf buffer1 buffer2 - location
  )

  (:init
    (at strawberry_0 table)
    (at strawberry_1 table)
    (buffer buffer1)
    (buffer buffer2)
    (buffer-free buffer1)
    (buffer-free buffer2)
    (clear strawberry_0)
    (clear strawberry_1)
    (goal-at strawberry_0 right_storage)
    (graspable strawberry_0)
    (graspable strawberry_1)
    (handempty)
    (safe strawberry_0)
    (safe strawberry_1)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target strawberry_0)
    (target strawberry_1)
  )

  (:goal
    (and
      (at strawberry_0 right_storage)
    )
  )
)
