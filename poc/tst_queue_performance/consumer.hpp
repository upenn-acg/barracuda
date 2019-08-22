#pragma once

class Consumer
{
public:
    enum { BUFFER_LEN = 16 * 1024 * 1024 };

    Consumer(PCHeader* dev_pcheader, PCRecord* dev_pcstart, size_t dev_pcbuffer_size) : _logged(0)
    {
        _dev_pcheader = dev_pcheader;
        _dev_pcstart = dev_pcstart;
        _dev_pcbuffer_size = (int)dev_pcbuffer_size;
        _buffer_len = min(_dev_pcbuffer_size, BUFFER_LEN);
        checkCudaErrors(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
        checkCudaErrors(cudaHostAlloc((void**)&_buffer, sizeof(PCRecord) * _buffer_len, cudaHostAllocPortable)); 
    }

    ~Consumer()
    {
        checkCudaErrors(cudaStreamDestroy(_stream));
        cudaFreeHost(_buffer);
    }

    void run(volatile bool* quit)
    {
        while(!*quit)
        {
            usleep(50000);
            consume_available();
        }
        consume_available();
    }

    long get_logged()
    {
        return _logged;
    }

private:
    void consume_available()
    {
        for(;;)
        {
            PCHeader pch;
            checkCudaErrors(cudaStreamSynchronize(_stream));
            checkCudaErrors(cudaMemcpyAsync(&pch, _dev_pcheader, 
                            sizeof(PCHeader), cudaMemcpyDeviceToHost, _stream));
            checkCudaErrors(cudaStreamSynchronize(_stream));
    
            DEBUGONLY(printf("** Read Head at: %i\n", pch.read_head);)
            DEBUGONLY(printf("** Write Head at: %i\n", pch.write_head);)
            DEBUGONLY(printf("** Tail at: %i\n", pch.tail);)

            int amount = min(pch.read_head - pch.tail, _buffer_len);
            if(amount == 0)
                break;
            DEBUGONLY(printf("** Reading %i records.\n", amount);)
 
            int err;
            PCRecord* start = _dev_pcstart + (pch.tail % _dev_pcbuffer_size);
//            checkCudaErrors(cudaMemcpyAsync(_buffer, start,
//                        sizeof(PCRecord) * amount, cudaMemcpyDeviceToHost, _stream));
            if(0 != (err = cudaMemcpyAsync(_buffer, start,
                        sizeof(PCRecord) * amount, cudaMemcpyDeviceToHost, _stream)))
            {
                fprintf(stderr, "cudaMemcpyAsync failed with %i:, start=%p end=%p tail=%i start=%p size=%i amount=%i\n", err, start,start + amount, pch.tail, _dev_pcstart, _dev_pcbuffer_size, amount);
                exit(1); 
            }

            checkCudaErrors(cudaStreamSynchronize(_stream));
            
            SlimFast::process_records(_buffer, amount);
            pch.tail += amount;
            _logged += amount;
        
            checkCudaErrors(cudaMemcpyAsync(&_dev_pcheader->tail, &pch.tail, 
                        sizeof(pch.tail), cudaMemcpyHostToDevice, _stream));
            checkCudaErrors(cudaStreamSynchronize(_stream));
        }    
        
    }

private:
    cudaStream_t _stream;
    PCHeader* _dev_pcheader;
    PCRecord* _dev_pcstart;
    PCRecord* _buffer;
    int  _buffer_len;
    int _dev_pcbuffer_size;
    long _logged;
};

