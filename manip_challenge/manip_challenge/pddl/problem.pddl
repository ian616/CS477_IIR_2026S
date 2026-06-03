(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    hammer strawberry_0 strawberry_1 - item
    left_storage right_storage bookshelf dynamic_buffer - location
  )

  (:init
    (at strawberry_0 table)
    (at strawberry_1 table)
    (buffer dynamic_buffer)
    (buffer-free dynamic_buffer)
    (clear strawberry_0)
    (clear strawberry_1)
    (goal-at hammer right_storage)
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
      (at hammer right_storage)
    )
  )
)
