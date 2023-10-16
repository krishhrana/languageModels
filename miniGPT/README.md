# languageModels
Implementation of various foundational models from scratch

### 1. tinyshakLM
`tinyshakLM` is a tiny (~5M) GPT-inspired model, trained on the entirety of Shakespeare's plays, designed to generate content reminiscent of Shakespeare's style.<br>

<img src="https://github.com/krishhrana/languageModels/blob/main/miniGPT/shakespeare.png" width="256"> <br>
It implements a character-level decoder-only transformer model with Masked Multi-Head Attention and it is trained on the 
[tiny-shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset. It also implements various decoding strategies such as controlling `temperature` and `top-p` for character-level language generation

![Loss Curve](https://github.com/krishhrana/languageModels/blob/main/miniGPT/tinyShakLM%20-%20loss_curve.png)

A sample text generated from the model: 
```
MARCIUS:
The fieuhereon is the world,
I will the heavens for thee the straight of you.

DUKE OF YORK:
Now, thou hast made between you moved
By this pretty to be banish'd, thy tale thy wife
As false for his friends as he shall the pale.

HENRY BOLINGBROKE:
But so that is done of garden from the man
Of his captive and their to heavy born.

DUKE OF YORK:
My gracious lord, that desire is such a doom
To give out a support a man of such shall be
chance to the crown.

HENRY BOLINGBROKE:
My lord, will I know the company.

KING RICHARD II:
So is the noble come to me hither fortune.

KING RICHARD II:
Sweet you be gone: I will the victory is father
With our title thou art not of his of this land
Which they do the singly art of thy chance:
They shall not die of thy lips on his sword,
Which will not be but to the state at them.

EDWARD:
Now will you be so doth the first the first,
Signior Hereford```
