#pragma once

class Consumer
{
public:
    enum { BUFFER_LEN = 1024 * 1024 };

    Consumer(PCHeader* dev_pcheader, PCRecord* dev_pcstart) 
    {
        _dev_pcheader = dev_pcheader;
        _dev_pcstart = dev_pcstart;
        checkCudaErrors(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
    
        _buffer = new PCRecord[BUFFER_LEN];
    }

    ~Consumer()
    {
        checkCudaErrors(cudaStreamDestroy(_stream));
        delete[] _buffer;
    }

    void run(volatile bool* quit)
    {
        while(!*quit)
        {
            sleep(1);
            consume_available();
        }
        consume_available();
    }

private:
    void consume_available()
    {
        PCHeader pch;
        checkCudaErrors(cudaMemcpyAsync(&pch, _dev_pcheader, 
                        sizeof(PCHeader), cudaMemcpyDeviceToHost, _stream));
        checkCudaErrors(cudaStreamSynchronize(_stream));
    
        for(;;)
        {
            printf("** Read Head at: %i\n", pch.read_head);
            printf("** Write Head at: %i\n", pch.write_head);
            printf("** Tail at: %i\n", pch.tail);

            int amount = pch.read_head - pch.tail;
            if(amount == 0)
                break;
            if(amount > BUFFER_LEN)
                amount = BUFFER_LEN;
            printf("** Reading %i records.\n", amount);
 
            checkCudaErrors(cudaMemcpyAsync(_buffer, _dev_pcstart + pch.tail, 
                        sizeof(PCRecord) * amount, cudaMemcpyDeviceToHost, _stream));
            checkCudaErrors(cudaStreamSynchronize(_stream));
            
            SlimFast::process_records(_buffer, amount);
            pch.tail += amount;
        }    
        
        checkCudaErrors(cudaMemcpyAsync(&_dev_pcheader->tail, &pch.tail, 
                        sizeof(pch.tail), cudaMemcpyHostToDevice, _stream));
        checkCudaErrors(cudaStreamSynchronize(_stream));
    }

private:
    cudaStream_t _stream;
    PCHeader* _dev_pcheader;
    PCRecord* _dev_pcstart;
    PCRecord* _buffer;
};

