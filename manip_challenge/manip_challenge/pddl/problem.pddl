(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    coke_can hammer_0 meat_can strawberry - item
    left_storage right_storage bookshelf dynamic_buffer - location
  )

  (:init
    (at hammer_0 table)
    (buffer dynamic_buffer)
    (buffer-free dynamic_buffer)
    (clear hammer_0)
    (goal-at hammer_0 right_storage)
    (graspable hammer_0)
    (handempty)
    (safe hammer_0)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target hammer_0)
  )

  (:goal
    (and
      (at hammer_0 right_storage)
    )
  )
)
