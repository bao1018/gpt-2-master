import tensorflow as tf
import tensorflow_hub as hub

MODEL_NAME = "use"
VERSION = 1
SERVE_PATH = "./models/{}/{}".format(MODEL_NAME, VERSION)

with tf.Graph().as_default():
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    text = tf.placeholder(tf.string, [None])
    embedding = embed(text)

    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    with tf.Session() as session:
        session.run(init_op)
        tf.saved_model.simple_save(
            session,
            SERVE_PATH,
            inputs={"text": text},
            outputs={"embedding": embedding},
            legacy_init_op=tf.tables_initializer(),
        )

