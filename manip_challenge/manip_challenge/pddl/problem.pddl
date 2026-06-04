(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    banana_0 coke_can_0 hammer_0 meat_can_0 - item
    left_storage right_storage bookshelf dynamic_buffer - location
  )

  (:init
    (at banana_0 table)
    (at coke_can_0 table)
    (at hammer_0 table)
    (at meat_can_0 table)
    (blocks meat_can_0 banana_0)
    (buffer dynamic_buffer)
    (buffer-free dynamic_buffer)
    (clear banana_0)
    (clear coke_can_0)
    (clear hammer_0)
    (clear meat_can_0)
    (goal-at hammer_0 right_storage)
    (graspable banana_0)
    (graspable coke_can_0)
    (graspable hammer_0)
    (graspable meat_can_0)
    (handempty)
    (obstacle banana_0)
    (obstacle coke_can_0)
    (obstacle meat_can_0)
    (safe banana_0)
    (safe coke_can_0)
    (safe hammer_0)
    (safe meat_can_0)
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
